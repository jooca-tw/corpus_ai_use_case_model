"""步驟 6：用四選一選擇題比較模型（例如原始模型 vs 最終模型）。

做法：把題目和 A～D 選項放進對話，看模型下一個字選 A、B、C、D 的分數，取最高者當答案（不讓模型自由生成）。
為了避免模型偏好某個字母，每題會輪換選項順序作答多次（4 或 8 種，每個選項出現在每個位置的次數相同），
再把各選項的分數平均後判斷。

題目來源二選一：
  --data my_test.jsonl         自己的題目，每行 {"id": "q1", "question": "...", "choices": ["甲", "乙", "丙", "丁"], "answer": "A"}
  --tmmluplus all 從 Hugging Face 下載公開的 TMMLU+ 測驗題（all＝全部科目，也可以只列幾個科目）

評 CPT 的某個訓練點（挑訓練點用，請用另一份驗證題，不要用最後報分數的測驗題）：
  python scripts/eval_mcq.py --model google/gemma-4-31B-it --adapters work/cpt/checkpoint-500 --data data/my_dev.jsonl --out work/eval/ckpt500

用法：
  python scripts/eval_mcq.py --model google/gemma-4-31B-it --tmmluplus all --out work/eval/base
  python scripts/eval_mcq.py --model work/final_model     --tmmluplus all --out work/eval/final
  （all 約兩萬題，評測較久；可以只列幾個科目，例如 --tmmluplus geography_of_taiwan taiwanese_hokkien，或先加 --limit 200 試跑）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, load_tmmluplus, read_jsonl, require_chat_template  # noqa: E402

# 位置平衡的排列：4 種＝循環位移；8 種＝再加反向循環
PERMS = {
    4: ["ABCD", "BCDA", "CDAB", "DABC"],
    8: ["ABCD", "BCDA", "CDAB", "DABC", "DCBA", "CBAD", "BADC", "ADCB"],
}


def load_items(args):
    items = []
    if args.data:
        for i, r in enumerate(read_jsonl(args.data)):
            ch = r["choices"]
            if len(ch) != 4:
                raise SystemExit(f"{args.data} 第 {i + 1} 題不是 4 個選項")
            items.append({"id": r.get("id", str(i)), "question": r["question"],
                          "A": ch[0], "B": ch[1], "C": ch[2], "D": ch[3],
                          "answer": str(r["answer"]).strip().upper(), "group": r.get("group", "")})
    if args.tmmluplus:
        items += load_tmmluplus(args.tmmluplus)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("沒有題目：請給 --data 或 --tmmluplus")
    return items


def build_prompt(item, tok):
    q = (f"題目：{item['question']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案：")
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_item(item, model, tok, letter_ids, perms):
    total = {k: 0.0 for k in "ABCD"}
    for perm in perms:
        # perm[i] ＝ 原本的哪個選項放在第 i 個位置
        shown = dict(item)
        for pos, orig in zip("ABCD", perm):
            shown[pos] = item[orig]
        enc = tok(build_prompt(shown, tok), return_tensors="pt", add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[0, -1].float()
        for pos, orig in zip("ABCD", perm):
            total[orig] += logits[letter_ids[pos]].item()
    avg = {k: v / len(perms) for k, v in total.items()}
    return max(avg, key=avg.get), avg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="模型名稱或路徑（原始模型或 merge_model.py 的輸出）")
    ap.add_argument("--adapters", nargs="*", default=[], help="（選用）不先合併，直接依序掛上 adapter 評測")
    ap.add_argument("--data")
    ap.add_argument("--tmmluplus", nargs="*")
    ap.add_argument("--perms", type=int, default=4, choices=[1, 4, 8], help="每題輪換幾種選項順序")
    ap.add_argument("--limit", type=int, default=None, help="試跑用：只評前 N 題")
    ap.add_argument("--load_in_4bit", action="store_true", help="顯存放不下時用 4-bit 載入（分數會和 bf16 有些微差異）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    items = load_items(args)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    require_chat_template(tok, args.model)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.model, device_map=device_map, for_training=False, load_in_4bit=args.load_in_4bit)
    for a in args.adapters:
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    model.eval()

    letter_ids = {}
    for L in "ABCD":
        ids = tok.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"tokenizer 把 {L!r} 切成 {len(ids)} 個 token，這支程式不適用")
        letter_ids[L] = ids[0]
    perms = PERMS.get(args.perms, ["ABCD"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    correct, by_group = 0, {}
    try:
        from tqdm import tqdm
        it = tqdm(items, desc="評測")
    except ImportError:
        it = items
    with open(out / "per_item.jsonl", "w", encoding="utf-8") as f:
        for item in it:
            pred, avg = score_item(item, model, tok, letter_ids, perms)
            ok = pred == item["answer"]
            correct += ok
            g = by_group.setdefault(item["group"] or "all", [0, 0])
            g[0] += ok
            g[1] += 1
            f.write(json.dumps({"id": item["id"], "group": item["group"], "answer": item["answer"],
                                "pred": pred, "correct": ok, "avg_logits": avg}, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "adapters": args.adapters, "n": len(items), "perms": perms,
               "accuracy": round(100 * correct / len(items), 2),
               "by_group": {k: round(100 * c / n, 2) for k, (c, n) in by_group.items()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
