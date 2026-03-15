# AI Capstone Project #1 — 進度記錄 (2026/03/15)

## 整體進度

| Step | 項目 | 狀態 |
|------|------|------|
| 1 | Pipeline Skeleton | **Done** |
| 2 | 資料收集與清理 | **Done** |
| 3 | 訓練與評估 | **Done** |
| 4 | 實驗 | **Done** |
| 4.5 | BERT embeddings + 模型升級 | **Done** (3/15 新增) |
| 4.6 | 應用場景：餐廳分析工具 | **Done** (3/15) |
| 5 | 報告 + 繳交 | **進行中** (deadline: 4/3) |

---

## Step 1: Pipeline — Done (3/14 晚)

### 專案結構
```
AI_HW1/
├── data/
│   ├── raw/reviews.csv              # 6,547 筆原始評論
│   ├── processed/
│   │   ├── reviews_clean.csv        # 6,509 筆清理後
│   │   ├── X_train.npy / X_test.npy # TF-IDF+SVD 特徵
│   │   ├── X_train_bert.npy / X_test_bert.npy  # BERT embeddings (1024維)
│   │   ├── tfidf.pkl / svd.pkl      # TF-IDF + SVD 轉換器
│   │   └── y_train*.npy / y_test*.npy
│   └── dummy/reviews.csv            # 300 筆假資料
├── src/
│   ├── collect.py           # Playwright Google Maps 爬蟲
│   ├── preprocess.py        # jieba + TF-IDF + TruncatedSVD
│   ├── train.py             # LR + SVM + 5-fold CV
│   ├── evaluate.py          # save_all_plots() 一鍵產圖
│   ├── experiments.py       # 5 個實驗封裝
│   ├── bert_baseline.py     # BERT zero-shot (GPU)
│   ├── bert_features.py     # BERT embedding 提取 (GPU) ← 新增
│   ├── train_bert_svm.py    # BERT+LR/SVM 訓練+比較 ← 新增
│   ├── analyze_restaurant.py # 餐廳分析應用工具 ← 新增
│   └── clean.py             # 資料清理
├── results/
│   ├── models/
│   │   ├── lr_model.pkl / svm_model.pkl          # TF-IDF 模型
│   │   ├── lr_bert_model.pkl / svm_bert_model.pkl # BERT 3-class 模型
│   │   └── lr_bert_binary_model.pkl               # BERT binary 模型 (最佳) ← 新增
│   ├── figures/              # 22+ 張圖 (含 BERT_SVM/BERT_LR 的圖)
│   ├── tables/               # 10 個 LaTeX 表格
│   └── analysis/             # 餐廳分析結果 ← 新增
├── scrape_full.py            # 多輪爬取腳本 ← 新增
├── run_pipeline.py           # 一鍵跑全 pipeline
├── generate_dummy_data.py
├── requirements.txt
└── README.md
```

### 驗證
- [x] `preprocess.py`: jieba + user_dict.txt 正確載入，TF-IDF + SVD 不報錯
- [x] `evaluate.py`: `save_all_plots()` 產出 confusion matrix / ROC / PR curve
- [x] `requirements.txt`: 完整依賴清單
- [x] 整個 pipeline 用 dummy data 跑通

---

## Step 2: 資料收集 — Done (3/15)

### 爬蟲
- **工具**: Playwright + Chromium headless
- **來源**: Google Maps 台灣餐廳評論
- **搜尋查詢**: 45 組（台北/新北/台中/台南/高雄/新竹/嘉義/花蓮/宜蘭/基隆）
- **防封鎖**: 隨機間隔 2-5 秒、每 5 間餐廳長休息、隨機查詢順序

### 資料集統計
| 項目 | 數量 |
|------|------|
| 原始評論 | 6,547 筆 |
| 清理後 | 6,509 筆 |
| 餐廳數 | 218 間 |
| 地區數 | 18 個 |
| 正面 (4-5★) | 5,162 (79.3%) |
| 中性 (3★) | 664 (10.2%) |
| 負面 (1-2★) | 683 (10.5%) |

### 地區覆蓋
台北、新北、台中、台南、高雄、新竹、嘉義、基隆、宜蘭、花蓮、士林夜市、饒河街夜市、寧夏夜市、逢甲夜市、花園夜市、六合夜市、瑞豐夜市、一中街

### 清理規則
- 移除空白/NaN 評論
- 移除 < 5 字評論
- 移除不含中文的評論
- 移除完全重複評論
- 移除 rating=0（擷取失敗）
- 清理後移除 38 筆（99.4% 保留率）

---

## Step 3: 訓練與評估 — Done (3/15)

### 特徵
- TF-IDF: max_features=5000
- TruncatedSVD: 150 維, explained variance = 25%
- Train/test split: 80/20, stratified

### 模型結果

| Model | Accuracy | F1 Macro | Precision Macro | Recall Macro | AUROC |
|-------|----------|----------|-----------------|--------------|-------|
| **Logistic Regression** | 81.1% | 0.432 | 0.703 | 0.415 | 0.812 |
| **SVM (RBF)** | 81.9% | 0.447 | 0.509 | 0.448 | 0.795 |
| **BERT Zero-Shot** | 81.5% | 0.559 | — | — | 0.824 |

### 5-Fold CV 結果
- LR: accuracy=0.818±0.005, F1_macro=0.449±0.022
- SVM: accuracy=0.822±0.006, F1_macro=0.457±0.015

### 觀察
- Accuracy 高 (~82%) 但主要因為正面類別佔 79%
- F1 Macro 偏低，模型難以區分中性/負面
- BERT zero-shot 在 F1 Macro 上優於傳統 ML（0.559 vs 0.447）
- Confusion matrix 顯示大量中性/負面被誤判為正面

---

## Step 4: 實驗 — Done (3/15)

### Exp 1: Learning Curve
- 圖表已產出（`learning_curve_lr.png`, `learning_curve_svm.png`）
- LaTeX 表格已產出

### Exp 2: Class Balance
| Strategy | Accuracy | F1 Macro | Precision | Recall |
|----------|----------|----------|-----------|--------|
| Original | 0.811 | 0.432 | 0.703 | 0.415 |
| **Balanced weights** | **0.647** | **0.516** | **0.506** | **0.597** |
| SMOTE | 0.656 | 0.506 | 0.495 | 0.577 |

- `class_weight='balanced'` 犧牲 accuracy 但 F1 提升 19.5%
- SMOTE 效果類似 balanced weights

### Exp 3: SVD Dimensions
| SVD Dim | Accuracy | F1 Macro | Var Explained |
|---------|----------|----------|---------------|
| None (raw TF-IDF) | 0.820 | 0.459 | 100% |
| 50 | 0.811 | 0.416 | 12.1% |
| 100 | 0.817 | 0.443 | 19.4% |
| 150 | 0.818 | 0.449 | 25.2% |
| 200 | 0.818 | 0.454 | 29.9% |

- Cumulative Explained Variance 圖已產出
- 200 維也只解釋 30% variance，TF-IDF 矩陣非常稀疏
- Raw TF-IDF 略優於 SVD，但維度高（5000）

### Exp 4: Neutral Class Handling (亮點實驗)
| Scenario | N | Classes | Accuracy | F1 Macro |
|----------|---|---------|----------|----------|
| 3-class | 6,509 | 3 | 0.818 | 0.449 |
| Binary (移除中性) | 5,845 | 2 | 0.898 | 0.595 |
| **3星→負面** | **6,509** | **2** | **0.829** | **0.625** |
| 3星→正面 | 6,509 | 2 | 0.902 | 0.547 |

- **「3星歸負面」效果最好** (F1=0.625)，反映台灣消費者給 3 星通常是不滿意
- 移除中性 vs 保留：二分類大幅優於三分類
- 這個結果支持台灣消費文化中「3星 ≈ 不及格」的假設

### Exp 5: Data Augmentation
- Noise injection 結果已產出
- LaTeX 表格已產出

### 產出清單
- **16 張圖表** (`results/figures/`)
- **10 個 LaTeX 表格** (`results/tables/`)，可直接 `\input{}`
- **3 個 metrics JSON** (`results/metrics_*.json`)

---

## Step 4.5: BERT Embeddings + 模型升級 — Done (3/15 新增)

### 核心改進
用 BERT (`xlm-roberta-large-xnli`) 當**特徵提取器**（mean pooling → 1024 維向量），
再用我們訓練的 LR/SVM 做分類。這是 transfer learning，BERT 只負責 encode，
分類器是我們的貢獻。

### BERT Embedding 提取
- 模型：`joeddav/xlm-roberta-large-xnli`（與 zero-shot baseline 同一模型）
- 維度：1024（mean pooling of last hidden state）
- GPU：NVIDIA RTX 4090，batch_size=64
- 6,509 筆評論提取完成，已存 `X_train_bert.npy` / `X_test_bert.npy`

### 全模型評比（3-class）

| Model | Accuracy | F1 Macro | Precision | Recall |
|-------|----------|----------|-----------|--------|
| TF-IDF + SVM | 81.9% | 0.447 | 0.500 | 0.439 |
| TF-IDF + LR | 81.1% | 0.432 | 0.703 | 0.415 |
| BERT + SVM | 79.8% | 0.328 | 0.599 | 0.350 |
| **BERT + LR** | **84.2%** | **0.636** | **0.652** | **0.625** |
| BERT zero-shot | 81.5% | 0.559 | — | — |

### 全模型評比（binary: 3星歸負面）

| Model | Accuracy | F1 Macro | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **BERT + LR (binary)** | **89.2%** | **0.826** | **0.846** | **0.811** |
| BERT + LR (binary+balanced) | 84.9% | 0.792 | 0.771 | 0.826 |
| BERT + LinearSVC (binary+balanced) | 86.9% | 0.768 | 0.836 | 0.734 |

### 5-Fold CV (BERT + LR Binary — 最佳模型)
- accuracy: 0.8837 ± 0.0078
- f1_macro: 0.8132 ± 0.0101
- precision_macro: 0.8338 ± 0.0173
- recall_macro: 0.7975 ± 0.0080

### 重要發現
1. **BERT+LR 打敗 BERT zero-shot**（F1 0.636 vs 0.559）— supervised > zero-shot
2. **BERT+SVM 反而最差**（F1 0.328）— RBF kernel 不適合 1024 維 BERT 空間
3. **二分類 BERT+LR 是冠軍**（F1 0.826）— 結合 BERT embeddings + 「3星歸負面」
4. 故事線：TF-IDF→BERT embeddings 的演進，有清楚的實驗支撐

### 已存模型
- `results/models/lr_bert_model.pkl` — BERT+LR 3-class
- `results/models/svm_bert_model.pkl` — BERT+SVM 3-class
- `results/models/lr_bert_binary_model.pkl` — BERT+LR binary (最佳，F1=0.826)

### 新增圖表
- `confusion_matrix_BERT_SVM.png` / `BERT_LR.png`
- `roc_curve_BERT_SVM.png` / `BERT_LR.png`
- `pr_curve_BERT_SVM.png` / `BERT_LR.png`
- `metrics_BERT_SVM.json` / `metrics_BERT_LR.json`

---

## Step 4.6: 應用場景 — 餐廳分析工具（3/15 新增，進行中）

### 功能
`src/analyze_restaurant.py` — 給一個 Google Maps 餐廳 URL，自動：
1. 爬取該餐廳的所有評論（多輪爬取 + 去重）
2. 用 BERT+LR (binary) 預測每則評論的滿意度
3. 按週統計滿意度趨勢
4. 產出趨勢圖 + 圓餅圖 + 文字摘要 + 負面評論摘要

### 使用方式
```bash
# 預設用 BERT+LR binary (最佳模型)
python src/analyze_restaurant.py "https://maps.app.goo.gl/xxxxx"

# 指定其他模型
python src/analyze_restaurant.py "URL" --model svm
python src/analyze_restaurant.py "URL" --model bert-lr
python src/analyze_restaurant.py "URL" --model bert

# 分析已有 CSV
python src/analyze_restaurant.py data.csv

# 多輪全量爬取
python scrape_full.py
```

### 可用模型選項
| 選項 | 模型 | F1 | 說明 |
|------|------|-----|------|
| `bert-lr-binary` (預設) | BERT embed + LR (二分類) | 0.826 | 滿意/不滿意，最強 |
| `bert-lr` | BERT embed + LR (三分類) | 0.636 | 正面/中性/負面 |
| `svm` | TF-IDF + SVM | 0.447 | 輕量，不需 GPU |
| `lr` | TF-IDF + LR | 0.432 | 輕量，不需 GPU |
| `bert` | BERT zero-shot | 0.559 | 無訓練 baseline |

### 測試案例：老鐵醬平價鐵板燒
- **URL**: `https://maps.app.goo.gl/GnGWYuEcMzzWe3H27?g_st=il`
- **Google Maps 顯示**: 2,834 則（含純星等無文字）
- **實際有文字可分析**: ~1,400+ 則（約 50% 的評論是純星等沒寫文字）

### 多輪爬取策略（`scrape_by_star.py`） — 完成
為最大化覆蓋率，採用多排序方式多輪爬取 + 去重合併：

| Pass | 排序方式 | 爬到 | 新增 | 累計 |
|------|---------|------|------|------|
| 1 | 最新 | 1503 | +61 | 1366 |
| 2 | 最相關 | 1509 | +30 | 1396 |
| 3 | 最低評分 | 525 | +3 | 1399 |
| 4 | 最高評分 | 526 | +4 | 1403 |
| 5 | 最新(重跑) | 1540 | +24 | 1427 |
| 6 | 最相關(重跑) | 1550 | +16 | **1443** |

- **狀態**: 全部完成，已飽和
- 每輪結束自動存檔，crash 不會丟失進度

### 最終分析結果（1443 則，BERT+LR binary）
- 滿意 88.6% (1278) / 不滿意 11.4% (165)
- 平均星等：4.57
- 統計期間：2025/03 ~ 2026/03（共 13 個月）
- 2025/08 滿意率暴跌至 35.3%，花 2 個月恢復
- 最近趨勢上升：2026/03 滿意率 98.7%

### 負面評論分類（155 則）
| 類別 | 則數 | 最後出現 | 現況 |
|------|------|---------|------|
| 服務態度 | 41 | 2026/02 | 未解決 |
| 衛生環境 | 36 | 2026/02 | 大幅改善但未根除 |
| 食物品質 | 76 | 2026/02 | 未解決（太鹹/沒味道） |
| 出餐動線 | 34 | 2026/02 | 基本解決 |
| 刷評論（五星送飲料） | 27 | 2026/02 | 未解決，反效果 |
| 刺青員工 | 5 | 2026/02 | 未解決 |
| 包手員工 | 4 | 2026/02 | 未解決 |

### 程式修正（3/15 晚）
- `analyze_restaurant.py`: 日期從「週」改為「月」分組（Google Maps 日期為月粒度）
- 支援「上次編輯：X前」格式解析
- 趨勢圖加年份、圓餅圖加統計期間

### 產出檔案（在 `results/analysis/`）
- `raw_full_老鐵醬平價鐵板燒.csv` — 1443 筆全量原始評論
- `analysis_full_老鐵醬平價鐵板燒.csv` — 含預測情感（已用最終資料重跑）
- `trend_老鐵醬平價鐵板燒.png` — 每月趨勢圖（含年份）
- `pie_老鐵醬平價鐵板燒.png` — 情感圓餅圖（含統計期間）

---

## Step 5: 報告 — TODO

### 報告故事線（已規劃）
1. **研究問題與動機** (~0.5 頁) — 台灣美食評論情感分類 + 實際應用場景
2. **資料集文件** (~2 頁) — 6509 筆、爬蟲流程、EDA
3. **方法描述** (~2 頁) — 三階段演進：
   - Stage 1: TF-IDF + SVD → SVM/LR (F1=0.447)
   - Stage 2: BERT embeddings → LR (F1=0.636, +42%)
   - Stage 3: BERT embeddings → LR binary (F1=0.826, +85%)
   - Data Flow Diagram
4. **實驗與結果** (~3 頁) — 6 個實驗（原 5 個 + 特徵比較/模型總評比）
5. **應用展示** (~0.5 頁) — 餐廳分析工具 demo
6. **討論** (~1.5 頁) — BERT+LR > BERT zero-shot、二分類 vs 三分類、限制
7. **參考文獻** + **附錄程式碼**

### 待做
- [ ] 跑 EDA notebook 產出資料集文件圖表
- [ ] 生成 LaTeX 專案壓縮檔 → 上傳 Overleaf
- [ ] 撰寫報告各章節（max 10 頁）
- [ ] 上傳資料集 + 程式碼到 GitHub
- [ ] 透過 E3 繳交 PDF

---

## 技術環境
- Python 3.9 + CUDA 12.1
- 2x NVIDIA RTX 4090 (24GB each)
- scikit-learn, jieba, transformers, Playwright
- BERT model: `joeddav/xlm-roberta-large-xnli`

## 關鍵提醒
1. **最佳模型是 BERT+LR binary** (F1=0.826)，預設用這個
2. **報告故事線是三階段演進**：TF-IDF → BERT embeddings → binary
3. **應用場景**：餐廳老闆用工具分析自家 Google Maps 評論
4. **「3星歸負面」是亮點**：台灣消費文化 insight
5. **我們的貢獻**：BERT 當特徵提取器 + 我們訓練的 LR 分類器（transfer learning）
6. **全量爬取用** `scrape_full.py`（多輪 + 去重）
