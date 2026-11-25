"""步驟 3、5：用 LoRA 訓練（CPT 或 SFT 共用這一支）。

--mode cpt：讀 prepare_cpt_data.py 的輸出，整段文字都計算 loss（讓模型多讀文章）。
--mode sft：讀 prepare_sft_data.py 的輸出，只對回答計算 loss（讓模型學會照指示回答）。

原始權重不會改動，只會另外存一份 LoRA adapter（out/final_adapter）。

單卡：
  python scripts/train.py --mode cpt --model google/gemma-4-31B-it --data work/cpt_data/data --out work/cpt
多卡（例如 4 張）：
  torchrun --nproc_per_node=4 scripts/train.py --mode cpt ...
顯存不夠：加 --load_in_4bit（QLoRA）。
SFT 要接在 CPT 之後：用 --base_adapter 指定 CPT adapter（訓練前先在記憶體裡合併），
  或把 --model 換成 merge_model.py 合併好的 CPT 模型（用 --load_in_4bit 時只能這樣做）。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dist_info, load_model, lora_targets, seed_everything  # noqa: E402

# 兩種模式的預設值＝我們實際使用的設定（每步 10／20 筆＝5 張卡 × 每卡 1 筆 × 累積 2／4 步）
DEFAULTS = {
    "cpt": dict(lr=1e-5, lora_r=64, lora_alpha=128, global_batch=10),
    "sft": dict(lr=1e-4, lora_r=16, lora_alpha=32, global_batch=20),
}


class Collator:
    def __init__(self, mode, pad_id, add_mm):
        self.mode, self.pad_id, self.add_mm = mode, pad_id, add_mm

    def __call__(self, feats):
        if self.mode == "cpt":
            ids = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
            out = {"input_ids": ids, "labels": ids.clone()}
        else:
            L = max(len(f["input_ids"]) for f in feats)
            pad = lambda xs, v: list(xs) + [v] * (L - len(xs))  # noqa: E731
            out = {
                "input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in feats]),
                "labels": torch.tensor([pad(f["labels"], -100) for f in feats]),
                "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in feats]),
            }
        if self.add_mm:
            # Gemma 多模態模型訓練時要求這些欄位；全 0＝全部是文字（Gemma 4 用 mm_token_type_ids，Gemma 3 用 token_type_ids）
            out["mm_token_type_ids"] = torch.zeros_like(out["input_ids"])
            out["token_type_ids"] = torch.zeros_like(out["input_ids"])
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cpt", "sft"], required=True)
    ap.add_argument("--model", required=True, help="底模名稱或路徑")
    ap.add_argument("--data", required=True, help="prepare_*_data.py 輸出的 data 目錄")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_adapter", default=None, help="（SFT 用）先合併這個 CPT adapter 再訓練")
    ap.add_argument("--load_in_4bit", action="store_true", help="QLoRA：4-bit 載入底模，省顯存")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lora_r", type=int)
    ap.add_argument("--lora_alpha", type=int)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--global_batch", type=int, help="每步總共幾筆（預設 CPT 10、SFT 20），會依卡數自動換算累積步數")
    ap.add_argument("--batch_size", type=int, default=1, help="每張卡每次放幾筆")
    ap.add_argument("--grad_accum", type=int, default=None, help="直接指定累積幾步（指定後就不看 --global_batch）")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=None, help="最多保留幾個訓練點（預設全部保留）")
    ap.add_argument("--max_steps", type=int, default=-1, help="試跑用：只跑幾步")
    ap.add_argument("--limit", type=int, default=None, help="試跑用：只用前 N 筆資料")
    ap.add_argument("--resume", action="store_true", help="從 out 裡最新的訓練點接著跑（沒有就從頭開始）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for k, v in DEFAULTS[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.load_in_4bit and args.base_adapter:
        raise SystemExit("--load_in_4bit 不能和 --base_adapter 一起用：請先用 merge_model.py 把 CPT 合併成完整模型，"
                         "再把 --model 指向合併後的資料夾")

    from datasets import load_from_disk
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    local_rank, world_size, is_main = dist_info()
    seed_everything(args.seed)
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    per_step = args.batch_size * world_size
    if args.grad_accum is None:
        args.grad_accum = max(1, math.ceil(args.global_batch / per_step))
    eff = per_step * args.grad_accum
    if is_main and args.grad_accum and eff != args.global_batch:
        print(f"[train] 注意：{world_size} 張卡 × 每卡 {args.batch_size} 筆 × 累積 {args.grad_accum} 步＝每步 {eff} 筆"
              f"（目標 {args.global_batch} 筆）")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_from_disk(args.data)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    need = {"input_ids"} if args.mode == "cpt" else {"input_ids", "labels", "attention_mask"}
    if not need <= set(ds.column_names):
        raise SystemExit(f"{args.data} 欄位是 {ds.column_names}，不像 {args.mode} 的資料；--mode 是否填錯？")

    resume = None
    if args.resume:
        resume = get_last_checkpoint(str(out)) if out.is_dir() else None
        if is_main:
            print(f"[train] 從 {resume} 接著跑" if resume else "[train] 還沒有任何訓練點，從頭開始")

    device_map = {"": local_rank} if torch.cuda.is_available() else None
    model, multimodal = load_model(args.model, load_in_4bit=args.load_in_4bit, device_map=device_map)
    model.config.use_cache = False

    if args.base_adapter:
        if is_main:
            print(f"[train] 先合併 {args.base_adapter}")
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=False).merge_and_unload()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=lora_targets(multimodal), bias="none"))
    if is_main:
        model.print_trainable_parameters()
        print(f"[train] mode={args.mode} 資料 {len(ds):,} 筆、{world_size} 張卡、每步 {eff} 筆、"
              f"約 {max(1, math.ceil(len(ds) * args.epochs / eff)):,} 步；lr={args.lr} r={args.lora_r}")

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=False,  # 上面已經開了
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=multimodal if world_size > 1 else None,
        # 多卡時不要把第 1 張卡的 buffer 廣播給其他卡。Gemma 4、Gemma 3 的位置編碼 buffer 在每張卡上的排列順序可能不同
        # （由 Python 的隨機雜湊決定），廣播時會錯位，讓其他卡的位置編碼壞掉、loss 偏高。這些 buffer 都是固定值，不需要同步。
        ddp_broadcast_buffers=False,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(args.mode, tok.pad_token_id, multimodal))
    # Gemma 4 的 forward 接受額外參數，Trainer 會因此不把 loss 除以累積步數，畫面上的 loss 會被放大好幾倍。
    # 這裡關掉，讓顯示的 loss 是正常的每個 token 平均值。
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=resume)

    if is_main:
        final = out / "final_adapter"
        trainer.model.save_pretrained(str(final))
        tok.save_pretrained(str(final))
        print(f"[train] 完成 → {final}")


if __name__ == "__main__":
    main()
