# 台灣主權 AI 模型訓練教學（CPT＋SFT）

這份教學說明我們怎麼把一個公開的大語言模型（Google Gemma-4-31B-it）進行後續訓練/微調，並附上相關程式。照著這邊的步驟並換上準備好的語料，就能訓練出自己的模型。

> 註：若使用的語料和硬體不同，訓練出來的模型、表現、方法也會有差異。

---

## 整體做法

目標是讓模型更熟悉台灣的語言、文化、歷史與在地知識，作為打造台灣主權 AI 模型的基礎。

```
 公開模型            ① 準備資料        ② CPT 持續預訓練       ③ SFT 指令微調        ④ 合併            ⑤ 評測
Gemma-4-31B-it ──▶   台灣在地語料   ──▶  多讀台灣在地文章  ──▶ 學會照指示回答 ──▶ 最終模型 ──▶ 和原始模型比較
                   清理、和測驗題去重                                            （一組權重）
```

| 步驟 | 做什麼 | 為什麼 |
|---|---|---|
| ① 準備資料 | 收集台灣在地的文章（華語，以及台語等本土語言）和中英對照文字；清理後和測驗題比對，重疊的整篇剔除 | 讓模型多接觸台灣的用語與知識；資料較少的語言（例如本土語言）可以在訓練中重複幾次加重；加入中英對照文字，避免英文等原有能力退步；去重是為了不讓模型先看過考題 |
| ② CPT（持續預訓練） | 讓模型大量「閱讀」上面的文章 | 補充台灣在地用語、本土語言與在地知識 |
| ③ SFT（指令微調） | 用「問題－回答」對話資料訓練 | 讓模型學會照指示、照格式回答 |
| ④ 合併 | 原始模型＋CPT＋SFT 合成一個模型 | 方便使用與部署 |
| ⑤ 評測 | 用選擇題比較原始模型與最終模型 | 確認台灣相關的知識與語言能力有進步，原本的能力沒有退步（想確認英文等其他能力，可以另外準備測驗題） |

CPT 和 SFT 都用 **LoRA**：原始模型的權重不動，只另外訓練一小組附加權重（adapter），所需的顯存和時間都少很多。

---

## 環境需求

### 硬體

| 你的硬體 | 建議做法 | 說明 |
|---|---|---|
| 1 張或多張 96GB 顯卡 | 31B 模型、一般 LoRA（多卡可同時訓練，速度較快） | 實測單卡顯存峰值約 79GB；**我們的環境**：多張 96GB 顯卡 |
| 1 張 48～80GB 顯卡 | 31B 模型＋`--load_in_4bit`（QLoRA） | 實測顯存峰值約 43GB（CPT、序列長度 2048）；時間會比多卡長很多 |
| 1 張 24GB 顯卡 | 改用小模型（例如 `google/gemma-4-E4B-it`）＋`--load_in_4bit`，序列長度改 1024（`prepare_cpt_data.py --seq_len 1024`） | 小模型的中文與在地知識底子較弱，效果有限，適合練習流程 |

表中的建議都是粗估；正式跑之前請先用 `--max_steps 20` 試跑幾步，看顯存與速度。

磁碟（以 31B 為例）：底模約 60GB；CPT 每 500 步存一個訓練點，每個約 5.5GB（預設全部保留，可用 `--save_total_limit` 限制）；合併後的最終模型約 60GB；用 QLoRA 時還會多一份合併過 CPT 的模型（約 60GB）。合併時記憶體（RAM）至少要約 70GB。

### 軟體

- Python 3.11
- 先依自己的 CUDA 版本安裝 PyTorch（https://pytorch.org），再：

```bash
pip install -r requirements.txt
```

底模會從 Hugging Face 自動下載。底模請用**有對話格式的指令版**（名稱通常帶 `-it` 或 `-Instruct`）；需要先同意授權的模型（例如 Llama）請先 `huggingface-cli login`。使用前請自行確認底模的授權條款。

---

## 快速試跑（確認環境）

用 `examples/` 裡的幾筆範例資料和小模型（`google/gemma-4-E4B-it`），每階段只跑 3 步，確認整個流程能跑完：

```bash
DEMO=1 bash run_all.sh
```

- 第一次會下載約 16GB 的小模型，合併時再寫出約 16GB，請預留約 35GB 磁碟。
- 範例資料只是示範格式，不是真的訓練資料，訓練出來的模型沒有意義。

---

## 正式訓練：一步一步來

以下指令都在 repo 根目錄執行。

### 1. 準備資料

**CPT 語料**：jsonl 檔，每行一篇文章（或一段文字）。很長的文件（例如整本書）建議先切成段落，一段一行。

```json
{"text": "臺灣的廟宇常見於街頭巷尾，許多地方的信仰活動也和農業節氣有關。"}
```

建議依語言或來源分成幾個檔，例如：台灣在地的華語文章、台語等本土語言文章、中英對照（中英對照可以把英文和中文用空行接在同一個 text 裡），方便之後分別調整比重。

**SFT 資料**：jsonl 檔，每行一段對話，最後一則必須是 `assistant` 的回答。只有最後一則回答會被訓練；多輪對話如果每一輪都要訓練，請拆成多筆。

```json
{"messages": [{"role": "user", "content": "請用一句話介紹臺灣的夜市。"}, {"role": "assistant", "content": "臺灣的夜市是……"}]}
```

**測驗題**：jsonl 檔，每行一題四選一。建議準備兩份：**驗證題**（dev，訓練過程中挑訓練點用）和**測驗題**（test，最後報分數用，只評一次）。

```json
{"id": "q1", "question": "臺灣最高的山是哪一座？", "choices": ["雪山", "玉山", "合歡山", "阿里山"], "answer": "B"}
```

> 測驗題絕對不能出現在訓練資料裡，否則分數會失真。下一步的去重只能擋掉和題目相同的段落，題目很短時擋不住；最好在收集資料時就分開。

### 2. 把 CPT 語料轉成訓練格式

```bash
python scripts/prepare_cpt_data.py --model google/gemma-4-31B-it \
  --input data/mandarin.jsonl data/local_lang.jsonl:3 data/en_zh.jsonl \
  --decon_files data/my_dev.jsonl data/my_test.jsonl --decon_tmmluplus all \
  --out work/cpt_data
```

- 檔名後面的 `:3` 表示這個檔在訓練中重複 3 次（資料少的語言，例如本土語言，可以加重）。
- `--decon_files`、`--decon_tmmluplus`：和測驗題比對（中文連續 13 字、英文連續 13 個詞相同；題目本身不到 13 字時改成整句比對），重疊的文章整篇剔除；剔除了哪些寫在 `work/cpt_data/stats.json`。`--decon_tmmluplus all` 會和 TMMLU+ 全部科目比對，含有常見考題句（例如「下列敘述何者正確」）的文章也會被剔除，請看一下 stats.json 剔除的量是否合理；也可以只列要評測的科目。
- `--seq_len`：每條訓練序列的長度，預設 2048。語料太少、湊不滿一條時會提示。
- `--add_bos`（選用）：在每條序列開頭加上開頭符號 `<bos>`。我們當時沒有加，所以預設不加；Gemma 系列模型加上可能會比較好，可以自行嘗試。

### 3. CPT

```bash
# 單卡
python scripts/train.py --mode cpt --model google/gemma-4-31B-it --data work/cpt_data/data --out work/cpt
# 多卡（例如 4 張）
torchrun --nproc_per_node=4 scripts/train.py --mode cpt --model google/gemma-4-31B-it --data work/cpt_data/data --out work/cpt
```

- 預設值就是我們用的設定：LoRA rank 64、學習率 1e-5、每步 10 筆（程式會依卡數自動換算）、資料讀一遍。
- 顯存不夠就加 `--load_in_4bit`；中斷了加 `--resume` 接著跑。
- 每 500 步存一個訓練點（`work/cpt/checkpoint-*`），最後的結果在 `work/cpt/final_adapter`。

**（選用）挑訓練點**：可以把各訓練點拿去評測驗證題，挑表現最好的一個：

```bash
python scripts/eval_mcq.py --model google/gemma-4-31B-it --adapters work/cpt/checkpoint-500 \
  --data data/my_dev.jsonl --out work/eval/cpt_ckpt500
```

挑好之後，下面第 5、6 步的 `work/cpt/final_adapter` 換成那個訓練點（例如 `work/cpt/checkpoint-2500`）。挑訓練點請用驗證題，不要用最後報分數的測驗題，否則分數會偏高。

### 4. 把 SFT 資料轉成訓練格式

```bash
python scripts/prepare_sft_data.py --model google/gemma-4-31B-it --input data/sft.jsonl --out work/sft_data
```

程式會印出一筆範例，標出哪一段會被計分（只有回答部分）。

### 5. SFT（接在 CPT 之後）

```bash
python scripts/train.py --mode sft --model google/gemma-4-31B-it \
  --base_adapter work/cpt/final_adapter \
  --data work/sft_data/data --out work/sft
```

- `--base_adapter`：先把 CPT 的結果併進模型，再訓練新的 SFT adapter。
- 預設值就是我們用的設定：LoRA rank 16、學習率 1e-4、每步 20 筆、資料讀一遍。
- 如果用 `--load_in_4bit`，請先只合併 CPT：`python scripts/merge_model.py --model google/gemma-4-31B-it --adapters work/cpt/final_adapter --out work/cpt_merged`，再把 `--model` 改成 `work/cpt_merged`、拿掉 `--base_adapter`。

### 6. 合併成最終模型

```bash
python scripts/merge_model.py --model google/gemma-4-31B-it \
  --adapters work/cpt/final_adapter work/sft/final_adapter --out work/final_model
```

合併在 CPU 上做，不需要顯卡。

### 7. 評測

```bash
# 自己的測驗題：原始模型 vs 最終模型
python scripts/eval_mcq.py --model google/gemma-4-31B-it --data data/my_test.jsonl --perms 8 --out work/eval/base
python scripts/eval_mcq.py --model work/final_model      --data data/my_test.jsonl --perms 8 --out work/eval/final
# 公開的 TMMLU+（繁體中文多學科選擇題，多數題目來自台灣的考試）：原始模型 vs 最終模型
python scripts/eval_mcq.py --model google/gemma-4-31B-it --tmmluplus all --perms 4 --out work/eval/base_tmmlu
python scripts/eval_mcq.py --model work/final_model      --tmmluplus all --perms 4 --out work/eval/final_tmmlu
```

- 評測方式：看模型對 A～D 四個選項的分數，選最高的，不讓模型自由作答。
- 每題會輪換選項順序作答多次（`--perms`），讓每個選項出現在每個位置的次數相同，避免模型「偏好某個字母」影響分數。
- TMMLU+ 填 `all` 是全部 66 科、約兩萬題，每題又要輪換選項順序，31B 模型會評很久；可以只列幾個科目（例如 `geography_of_taiwan taiwanese_hokkien`），或先加 `--limit 200` 試跑估時間。
- 顯存放不下就加 `--load_in_4bit`。
- 結果在 `summary.json`（答對率）與 `per_item.jsonl`（每題的作答）。

也可以直接問模型幾個問題，看回答：

```bash
python scripts/chat.py --model work/final_model --prompt "請簡單介紹臺灣的三個傳統節日。"
```

---

## 一鍵執行

上面的步驟也可以用 `run_all.sh` 一次跑完（在 repo 根目錄執行）：

```bash
BASE_MODEL=google/gemma-4-31B-it NGPU=4 \
CPT_INPUTS="data/mandarin.jsonl data/local_lang.jsonl:3 data/en_zh.jsonl" \
SFT_INPUTS="data/sft.jsonl" TEST_FILE=data/my_test.jsonl \
bash run_all.sh
```

顯存不夠加 `QLORA=1`，序列長度用 `SEQ=1024`，試跑用 `MAX_STEPS=20`，中斷後用 `RESUME=1` 接著跑。其他變數寫在 `run_all.sh` 開頭。一鍵流程直接使用最後的訓練點，不做挑選。

一鍵流程預設會另外下載 TMMLU+ 的台灣地理、台灣閩南語兩科，用來去重和評測（`TMMLU="geography_of_taiwan taiwanese_hokkien"`）；要評全部科目設 `TMMLU=all`（評測時間會長很多），不需要就設 `TMMLU=""`。

---

## 小提醒

- **SFT 資料的語言**：模型回答時用什麼語言，主要取決於 SFT 資料。例如希望模型能用台語等本土語言回答，SFT 資料裡就要有該語言的對話。
- **生成設定**：我們曾用 repetition penalty 1.2 加 no_repeat_ngram 5，中文標點與換行被破壞；改用 1.05、不設 no_repeat 後正常。`chat.py` 用 1.05。
- **其他底模**：程式也支援一般的語言模型（例如 Llama、Qwen 的指令版），`--model` 換掉即可，但我們只實際訓練過 Gemma 4。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `scripts/prepare_cpt_data.py` | CPT 語料：加權、和測驗題去重、切成訓練序列 |
| `scripts/prepare_sft_data.py` | SFT 對話：套對話格式、只對回答計分 |
| `scripts/train.py` | LoRA 訓練（`--mode cpt` 或 `--mode sft`） |
| `scripts/merge_model.py` | 合併底模與 adapter |
| `scripts/eval_mcq.py` | 選擇題評測 |
| `scripts/chat.py` | 快速試問 |
| `run_all.sh` | 一次跑完整個流程 |
| `examples/` | 各種資料格式的範例（自行撰寫，僅示範格式） |
