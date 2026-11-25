"""各步驟共用的小工具：載入模型、決定 LoRA 要加在哪些層、讀 jsonl。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

# 只在文字解碼器的注意力層與 MLP 層加 LoRA（Gemma 4 多模態模型的影像／語音部分不動）。
GEMMA_MULTIMODAL_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)
PLAIN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def dist_info():
    """回傳 (local_rank, world_size, 是否主程序)。單卡直接執行時是 (0, 1, True)。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, world_size, local_rank == 0


def is_multimodal_gemma(model_path: str) -> bool:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    archs = getattr(cfg, "architectures", None) or []
    return model_type.startswith("gemma") and bool(archs) and "ConditionalGeneration" in archs[0]


def _unwrap_clippable_linear(model) -> int:
    """Gemma 4 會把部分 Linear 包在 Gemma4ClippableLinear 裡，PEFT 不認得。
    只拆掉「沒有實際作用」的外殼（use_clipped_linears=False），有作用的保留。只在訓練時呼叫。"""
    n = 0

    def _walk(parent):
        nonlocal n
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "Gemma4ClippableLinear" and not getattr(child, "use_clipped_linears", False):
                setattr(parent, name, child.linear)
                n += 1
            else:
                _walk(child)

    _walk(model)
    return n


def load_model(model_path: str, *, load_in_4bit: bool = False, device_map=None, for_training: bool = True):
    """載入底模。回傳 (model, is_multimodal)。

    - Gemma 4（多模態架構）用 AutoModelForImageTextToText，其他模型用 AutoModelForCausalLM。
    - load_in_4bit=True 時用 4-bit（QLoRA），給顯存較小的卡。
    """
    multimodal = is_multimodal_gemma(model_path)
    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if multimodal:
        from transformers import AutoModelForImageTextToText as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    model = Cls.from_pretrained(model_path, **kwargs)
    if multimodal and for_training:
        _unwrap_clippable_linear(model)
    return model, multimodal


def lora_targets(multimodal: bool):
    return GEMMA_MULTIMODAL_TARGETS if multimodal else PLAIN_TARGETS


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path} 第 {i} 行不是合法的 JSON：{e}")


def load_tmmluplus(subjects):
    """下載 TMMLU+（ikala/tmmluplus）的測驗題；subjects 可填科目名稱，或 ["all"] 代表全部科目。"""
    if not subjects:
        return []
    from datasets import get_dataset_config_names, load_dataset

    if list(subjects) == ["all"]:
        subjects = get_dataset_config_names("ikala/tmmluplus")
    rows = []
    for s in subjects:
        for i, r in enumerate(load_dataset("ikala/tmmluplus", s, split="test")):
            rows.append({"id": f"{s}-{i}", "question": r["question"], "A": r["A"], "B": r["B"],
                         "C": r["C"], "D": r["D"], "answer": r["answer"].strip().upper(), "group": s})
    return rows


def require_chat_template(tok, model_path):
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{model_path} 沒有對話格式（chat template）。請改用指令版的底模"
                         "（名稱通常帶 -it 或 -Instruct），或用已經合併好的模型資料夾。")


def seed_everything(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
