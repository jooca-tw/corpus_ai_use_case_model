"""步驟 5：把底模和 LoRA adapter 合併成一個完整模型（最終模型）。

adapter 會依序合併：先 CPT、再 SFT。
  python scripts/merge_model.py --model google/gemma-4-31B-it \
      --adapters work/cpt/final_adapter work/sft/final_adapter --out work/final_model

也可以只合併 CPT（給 --load_in_4bit 的 SFT 當底模用）：
  python scripts/merge_model.py --model google/gemma-4-31B-it --adapters work/cpt/final_adapter --out work/cpt_merged

合併在 CPU 上做，不需要 GPU，但記憶體要放得下整個模型（31B 至少約 70GB，建議預留更多）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    # 合併時不拆 Gemma4ClippableLinear，存出來的模型結構才會和原本一樣
    model, _ = load_model(args.model, device_map="cpu", for_training=False)
    for a in args.adapters:
        print(f"[merge] 合併 {a}")
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    out = Path(args.out)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(str(out))
    try:  # 多模態模型順便存 processor，之後用其他推論工具比較不會缺檔
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(str(out))
    except Exception:
        pass
    print(f"[merge] 完成 → {out}")


if __name__ == "__main__":
    main()
