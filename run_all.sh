#!/usr/bin/env bash
# 一次跑完整個流程：準備資料 → CPT → SFT → 合併 → 評測。請在 repo 根目錄執行。
#
# 先試跑（用 examples/ 的範例資料與小模型，每階段只跑 3 步，確認環境沒問題）：
#   DEMO=1 bash run_all.sh
# 正式訓練（換成自己的資料，依自己的硬體調整）：
#   BASE_MODEL=google/gemma-4-31B-it NGPU=4 \
#   CPT_INPUTS="data/mandarin.jsonl data/local_lang.jsonl:3 data/en_zh.jsonl" \
#   SFT_INPUTS="data/sft.jsonl" TEST_FILE=data/my_test.jsonl \
#   bash run_all.sh
#
# 可調的變數：
#   BASE_MODEL  底模（預設 google/gemma-4-31B-it；DEMO 預設 google/gemma-4-E4B-it）
#   NGPU        用幾張卡（預設 1）
#   QLORA=1     4-bit 載入（顯存不夠時）
#   SEQ         CPT 序列長度（預設 2048；顯存小可改 1024）
#   TMMLU       另外評測並去重的 TMMLU+ 科目（預設台灣地理、台灣閩南語兩科；all＝全部科目，評測時間會長很多；設成空字串就不用）
#   MAX_STEPS   每個訓練階段只跑幾步（試跑用）
#   RESUME=1    訓練中斷後接著跑
#   WORK        輸出目錄（預設 work）
#   PORT        多卡時的通訊埠（預設 29500）
set -euo pipefail
cd "$(dirname "$0")"

WORK=${WORK:-work}
NGPU=${NGPU:-1}
QLORA=${QLORA:-0}
SEQ=${SEQ:-2048}
TMMLU=${TMMLU-"geography_of_taiwan taiwanese_hokkien"}
EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max_steps "$MAX_STEPS")
[ "${RESUME:-0}" = 1 ] && EXTRA+=(--resume)

if [ "${DEMO:-0}" = 1 ]; then
  BASE_MODEL=${BASE_MODEL:-google/gemma-4-E4B-it}
  CPT_INPUTS=${CPT_INPUTS:-"examples/cpt_mandarin.jsonl examples/cpt_local_lang.jsonl:3 examples/cpt_en_zh.jsonl"}
  SFT_INPUTS=${SFT_INPUTS:-examples/sft.jsonl}
  TEST_FILE=${TEST_FILE:-examples/mcq.jsonl}
  SEQ=64; TMMLU=""; EXTRA+=(--max_steps 3)
  echo "== DEMO 模式：範例資料、序列長度 64、每階段只跑 3 步，只為確認流程能跑完 =="
else
  BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
  for v in CPT_INPUTS SFT_INPUTS TEST_FILE; do
    [ -n "${!v:-}" ] || { echo "請設定 $v（範例見 run_all.sh 開頭；只想試跑請用 DEMO=1）"; exit 1; }
  done
fi

Q=(); [ "$QLORA" = 1 ] && Q=(--load_in_4bit)
DECON=(--decon_files "$TEST_FILE"); [ -n "$TMMLU" ] && DECON+=(--decon_tmmluplus $TMMLU)
if [ "$NGPU" -gt 1 ]; then
  RUN=(torchrun --nproc_per_node="$NGPU" --master_port="${PORT:-29500}")
else
  RUN=(python)
fi

echo "== 1. 準備 CPT 資料 =="
python scripts/prepare_cpt_data.py --model "$BASE_MODEL" --input $CPT_INPUTS "${DECON[@]}" \
  --seq_len "$SEQ" --out "$WORK/cpt_data"

echo "== 2. CPT =="
"${RUN[@]}" scripts/train.py --mode cpt --model "$BASE_MODEL" --data "$WORK/cpt_data/data" \
  --out "$WORK/cpt" "${Q[@]}" "${EXTRA[@]}"

echo "== 3. 準備 SFT 資料 =="
python scripts/prepare_sft_data.py --model "$BASE_MODEL" --input $SFT_INPUTS --out "$WORK/sft_data"

echo "== 4. SFT（接在 CPT 之後）=="
if [ "$QLORA" = 1 ]; then
  # 4-bit 不能在記憶體裡合併 CPT，先合併成完整模型再訓練
  python scripts/merge_model.py --model "$BASE_MODEL" --adapters "$WORK/cpt/final_adapter" --out "$WORK/cpt_merged"
  SFT_BASE=(--model "$WORK/cpt_merged")
else
  SFT_BASE=(--model "$BASE_MODEL" --base_adapter "$WORK/cpt/final_adapter")
fi
"${RUN[@]}" scripts/train.py --mode sft "${SFT_BASE[@]}" --data "$WORK/sft_data/data" \
  --out "$WORK/sft" "${Q[@]}" "${EXTRA[@]}"

echo "== 5. 合併成最終模型 =="
python scripts/merge_model.py --model "$BASE_MODEL" \
  --adapters "$WORK/cpt/final_adapter" "$WORK/sft/final_adapter" --out "$WORK/final_model"

echo "== 6. 評測：原始模型 vs 最終模型 =="
for name in base final; do
  M=$BASE_MODEL; [ $name = final ] && M=$WORK/final_model
  python scripts/eval_mcq.py --model "$M" --data "$TEST_FILE" --perms 8 --out "$WORK/eval/$name/my_test" "${Q[@]}"
  if [ -n "$TMMLU" ]; then
    python scripts/eval_mcq.py --model "$M" --tmmluplus $TMMLU --perms 4 --out "$WORK/eval/$name/tmmluplus" "${Q[@]}"
  fi
done
echo "== 完成：最終模型在 $WORK/final_model，評測結果在 $WORK/eval/ =="
