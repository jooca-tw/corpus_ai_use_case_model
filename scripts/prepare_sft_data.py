"""步驟 3：把 SFT 用的對話整理成訓練格式。

輸入：jsonl，每行一段對話：
  {"messages": [{"role": "user", "content": "問題"}, {"role": "assistant", "content": "回答"}]}
  可以有多輪，也可以在最前面放 {"role": "system", ...}；最後一則必須是 assistant。
處理：套用底模的對話格式（chat template），只對「最後一則 assistant 回答」計算 loss，
      前面的提問與較早輪次的回答都不計分（多輪對話每輪都要訓練的話，請拆成多筆）；
      超過 --max_len 的部分截掉。
輸出：--out 目錄（HF datasets 格式）＋ stats.json。

用法：
  python scripts/prepare_sft_data.py --model google/gemma-4-31B-it \
      --input data/sft.jsonl --out work/sft_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, require_chat_template  # noqa: E402


def encode(messages, tok, max_len):
    """回傳 {"input_ids", "labels", "attention_mask"}；不合格回傳 None。

    提問部分的長度＝「完整對話」和「只到提問＋助理開頭」兩者 token 的共同前綴。
    有些模型（例如 Gemma-4-31B-it）的對話格式在助理開頭會多加一段空的思考區塊，
    但完整對話裡沒有；取共同前綴可以兩種格式都處理。
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    prefix_text = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    n = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        n += 1
    full_ids = full_ids[:max_len]
    labels = [-100] * min(n, len(full_ids)) + full_ids[n:]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="底模（只用它的 tokenizer 與對話格式）")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--show", type=int, default=1, help="印出前幾筆，確認哪些部分有計分")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    require_chat_template(tok, args.model)
    rows, skipped, truncated = [], 0, 0
    for path in args.input:
        for r in read_jsonl(path):
            e = encode(r.get("messages") or [], tok, args.max_len)
            if e is None:
                skipped += 1
                continue
            if len(e["input_ids"]) == args.max_len:
                truncated += 1
            rows.append(e)
    if not rows:
        raise SystemExit("沒有任何可用的對話（最後一則要是 assistant）")

    for e in rows[: args.show]:
        print("----- 範例（整段）-----")
        print(tok.decode(e["input_ids"]))
        print("----- 只有這部分計分 -----")
        print(tok.decode([t for t in e["labels"] if t != -100]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(out / "data"))
    stats = {"rows": len(rows), "skipped": skipped, "truncated_at_max_len": truncated, "max_len": args.max_len,
             "tokens": sum(len(e["input_ids"]) for e in rows),
             "loss_tokens": sum(sum(1 for t in e["labels"] if t != -100) for e in rows)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare_sft] 完成：{len(rows)} 筆（略過 {skipped}、截斷 {truncated}）→ {out / 'data'}")


if __name__ == "__main__":
    main()
