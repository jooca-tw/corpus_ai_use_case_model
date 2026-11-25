"""快速試問：對模型提一個問題，看它怎麼回答。

  python scripts/chat.py --model work/final_model --prompt "請簡單介紹臺灣的三個傳統節日。"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, require_chat_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    require_chat_template(tok, args.model)
    model, _ = load_model(args.model, device_map="auto" if torch.cuda.is_available() else "cpu",
                          for_training=False, load_in_4bit=args.load_in_4bit)
    for a in args.adapters:
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        # 我們用 repetition_penalty 1.2 加 no_repeat_ngram_size 5 時中文標點與換行被破壞；改用 1.05、不設 no_repeat
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.05)
    print(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
