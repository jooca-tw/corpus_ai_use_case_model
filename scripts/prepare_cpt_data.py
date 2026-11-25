"""步驟 2：把 CPT 用的文章整理成訓練格式。

輸入：一個或多個 jsonl 檔，每行 {"text": "一篇文章或一段文字"}。
      檔名後面可以加 :N 表示這個檔的內容在訓練中重複 N 次（資料少的語言可以加重），例如 local_lang.jsonl:3。
處理：
  1.（選用）和測驗題比對：文章裡只要有一段和題目相同（中日韓文字連續 13 字、英文等連續 13 個詞），
     整篇剔除，避免模型先看過題目。
  2. 依權重重複、打亂順序。
  3. 用底模的 tokenizer 切成 token，文章之間加結束符號，再接起來切成固定長度（預設 2048）的訓練序列；
     最後不滿一條序列的部分捨棄。
輸出：--out 目錄（data/ 是訓練資料）＋ stats.json。

用法：
  python scripts/prepare_cpt_data.py --model google/gemma-4-31B-it \
      --input data/mandarin.jsonl data/local_lang.jsonl:3 data/en_zh.jsonl \
      --decon_files data/my_test.jsonl --decon_tmmluplus all --out work/cpt_data
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_tmmluplus, read_jsonl  # noqa: E402

# 中日韓文字一個字算一個單位；英文、數字等連續字母算一個單位（一個詞）
_UNIT = re.compile(r"[㐀-鿿豈-﫿\U00020000-\U0003134f]|[^\W㐀-鿿豈-﫿\U00020000-\U0003134f]+",
                   re.UNICODE)
SHORT_MIN = 8  # 題目太短（少於 n 個單位）時改用「整題包含」比對，但至少要有這麼多單位，否則太容易誤刪


def _units(s: str):
    return _UNIT.findall(s.lower())


def _grams(units, n):
    return {"\x1f".join(units[i:i + n]) for i in range(len(units) - n + 1)}


class Decon:
    def __init__(self, n):
        self.n, self.grams, self.short = n, set(), set()
        self.items = self.uncheckable = 0

    def add(self, question, choices):
        self.items += 1
        covered = False
        for part in [question] + list(choices):
            u = _units(str(part))
            if len(u) >= self.n:
                self.grams |= _grams(u, self.n)
                covered = covered or part == question
        if not covered:
            u = _units(str(question))
            if len(u) >= SHORT_MIN:
                self.short.add("\x1f".join(u))
            else:
                self.uncheckable += 1

    def hit(self, text):
        u = _units(text)
        if self.grams and any(g in self.grams for g in _iter_grams(u, self.n)):
            return True
        if self.short:
            joined = "\x1f" + "\x1f".join(u) + "\x1f"
            return any(("\x1f" + s + "\x1f") in joined for s in self.short)
        return False


def _iter_grams(units, n):
    for i in range(len(units) - n + 1):
        yield "\x1f".join(units[i:i + n])


def parse_inputs(specs):
    out = []
    for spec in specs:
        path, weight = spec, 1
        m = re.match(r"^(.*):(\d+)$", spec)
        if m and not Path(spec).exists():
            path, weight = m.group(1), int(m.group(2))
        if not Path(path).exists():
            raise SystemExit(f"找不到輸入檔：{path}")
        out.append((path, weight))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="底模（只用它的 tokenizer）")
    ap.add_argument("--input", nargs="+", required=True, help="jsonl 檔，可加 :N 權重")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=2048, help="每條訓練序列的長度（顯存小可改 1024）")
    ap.add_argument("--decon_files", nargs="*", default=[], help="測驗題 jsonl（格式同 eval_mcq.py）")
    ap.add_argument("--decon_tmmluplus", nargs="*", default=[], help="也和這些 TMMLU+ 科目的測驗題比對")
    ap.add_argument("--decon_n", type=int, default=13, help="連續幾個字（英文為詞）相同就算重疊")
    ap.add_argument("--add_bos", action="store_true",
                    help="每條序列開頭加 <bos>（我們當時沒有加；Gemma 加了可能較好，未實測比較）")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"seq_len": args.seq_len, "add_bos": args.add_bos, "sources": {}}

    decon = None
    if args.decon_files or args.decon_tmmluplus:
        decon = Decon(args.decon_n)
        for f in args.decon_files:
            for row in read_jsonl(f):
                decon.add(row.get("question", ""), row.get("choices", []))
        for r in load_tmmluplus(args.decon_tmmluplus):
            decon.add(r["question"], [r["A"], r["B"], r["C"], r["D"]])
        stats["decon"] = {"files": args.decon_files, "tmmluplus": args.decon_tmmluplus, "items": decon.items,
                          "n": args.decon_n, "too_short_to_check": decon.uncheckable, "dropped_docs": 0,
                          "dropped_examples": []}
        print(f"[prepare_cpt] 測驗題 {decon.items} 題，比對單位 {args.decon_n}")
        if decon.uncheckable:
            print(f"[prepare_cpt] 注意：其中 {decon.uncheckable} 題太短，無法比對；這些題目請在收集資料時就和訓練資料分開")

    texts = []
    for path, weight in parse_inputs(args.input):
        kept, dropped, chars = 0, 0, 0
        for row in read_jsonl(path):
            t = (row.get("text") or "").strip()
            if not t:
                continue
            if decon and decon.hit(t):
                dropped += 1
                if len(stats["decon"]["dropped_examples"]) < 20:
                    stats["decon"]["dropped_examples"].append({"file": path, "text_head": t[:80]})
                continue
            texts.extend([t] * weight)
            kept += 1
            chars += len(t)
        stats["sources"][path] = {"docs": kept, "chars": chars, "weight": weight, "dropped_by_decon": dropped}
        if decon:
            stats["decon"]["dropped_docs"] += dropped
        print(f"[prepare_cpt] {path}: {kept} 篇、{chars:,} 字 ×{weight}" + (f"（和測驗題重疊剔除 {dropped} 篇）" if decon else ""))

    if not texts:
        raise SystemExit("沒有任何可用的文字")
    random.Random(args.seed).shuffle(texts)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    bos = tok.bos_token_id if args.add_bos else None

    def _tok(batch):
        return {"ids": tok(batch["text"], add_special_tokens=False)["input_ids"]}

    tok_ds = Dataset.from_dict({"text": texts}).map(
        _tok, batched=True, remove_columns=["text"], num_proc=max(1, args.num_proc), desc="切 token")
    total = sum(len(x) + 1 for x in tok_ds["ids"])
    body = args.seq_len - (1 if bos is not None else 0)
    n_seq = total // body
    if n_seq == 0:
        raise SystemExit(f"文字太少：共 {total} tokens，湊不滿一條 {args.seq_len} tokens 的序列；請加資料或把 --seq_len 改小")

    def _pack():
        buf, start = [], 0
        for row in tok_ds:
            buf.extend(row["ids"])
            buf.append(eos)
            while len(buf) - start >= body:
                chunk = buf[start:start + body]
                start += body
                yield {"input_ids": ([bos] + chunk) if bos is not None else chunk}
            del buf[:start]
            start = 0

    cache = out / ".cache"
    packed = Dataset.from_generator(_pack, cache_dir=str(cache))
    packed.save_to_disk(str(out / "data"))
    shutil.rmtree(cache, ignore_errors=True)
    stats.update(total_tokens=total, n_sequences=len(packed),
                 packed_tokens=len(packed) * args.seq_len, n_docs_after_weight=len(texts))
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare_cpt] 完成：{len(packed):,} 條序列（共 {total:,} tokens）→ {out / 'data'}")


if __name__ == "__main__":
    main()
