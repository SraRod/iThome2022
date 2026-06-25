# PyTorch 生態鏈實戰運用

以醫療影像為例，從資料準備到模型部署，實作完整的深度學習 pipeline。

本專案為 [iThome 2022 鐵人賽](https://ithelp.ithome.com.tw/users/20141304/ironman/5765) 30 天系列文的程式碼，使用 ChestMNIST 胸腔 X 光資料集進行 14 類疾病的多標籤分類，最佳結果達到 val_acc=0.95、val_auroc=0.78。

## 技術架構

| 類別 | 工具 |
|------|------|
| Framework | PyTorch, PyTorch Lightning |
| 醫療影像 | MONAI |
| Model | EfficientNet-B0 (timm, transfer learning) |
| 訓練優化 | LR Warmup, CosineAnnealing, Label Smoothing, Data Augmentation, Weight Decay, Lp Regularization |
| 實驗追蹤 | Weights & Biases |
| 模型部署 | ONNX, NVIDIA Triton Inference Server |
| 環境 | Docker, docker-compose |
| 資料集 | MedMNIST v2 (ChestMNIST) |

## 專案流程

```
make_dataset.py     資料下載與整理
       ↓
preprocess.py       資料前處理 / Data Augmentation
       ↓
  model.py          建構 EfficientNet-B0 Lightning Module
       ↓
  tuner.py          Learning Rate Finder
       ↓
  train.py          模型訓練 (WandB logging + checkpoint)
       ↓
evaluate.py         模型評估
       ↓
 export.py          匯出 ONNX 模型
       ↓
evaluate_with_triton.py   Triton Inference Server 推論驗證
```

## 專案結構

```
.
├── src/
│   ├── make_dataset.py          # 資料集準備
│   ├── preprocess.py            # 前處理與資料增強
│   ├── model.py                 # Multi-label Lightning Module
│   ├── tuner.py                 # LR Finder
│   ├── train.py                 # 訓練流程
│   ├── evaluate.py              # 模型評估
│   ├── export.py                # ONNX 匯出
│   └── evaluate_with_triton.py  # Triton 推論
├── deploy/
│   └── chestmnist_net/
│       └── config.pbtxt         # Triton 模型設定
├── hparams.yaml                 # 超參數設定
├── Dockerfile
├── docker-compose.yml           # 開發環境
└── docker-compose-triton.yml    # Triton 推論服務
```

## 快速開始

```bash
# 啟動開發環境
docker-compose up -d

# 資料準備 → 訓練 → 評估 → 匯出
python -m src.make_dataset --config hparams.yaml
python -m src.train --config hparams.yaml
python -m src.evaluate --config hparams.yaml
python -m src.export --config hparams.yaml

# 啟動 Triton 推論服務
docker-compose -f docker-compose-triton.yml up -d
python -m src.evaluate_with_triton --config hparams.yaml
```

## 系列文章

### 環境建置與資料準備

- [[Day 1] PyTorch 生態鏈實戰運用 - 系列文概要](https://github.com/SraRod/iThome2022/issues/1)
- [[Day 2] Containerized Development for Deep Learning](https://github.com/SraRod/iThome2022/issues/2)
- [[Day 3] MedMNIST v2](https://github.com/SraRod/iThome2022/issues/5)
- [[Day 4] Data Preparation for MedMNIST](https://github.com/SraRod/iThome2022/issues/8)
- [[Day 5] Dataloader with PyTorch and MONAI](https://github.com/SraRod/iThome2022/issues/10)

### 模型建構與訓練

- [[Day 6] Design a Model](https://github.com/SraRod/iThome2022/issues/12)
- [[Day 7] Model Training with PyTorch](https://github.com/SraRod/iThome2022/issues/14)
- [[Day 8] Model Validation](https://github.com/SraRod/iThome2022/issues/16)
- [[Day 9] Deep Learning with Configuration](https://github.com/SraRod/iThome2022/issues/18)
- [[Day 10] Pytorch-Lightning](https://github.com/SraRod/iThome2022/issues/20)
- [[Day 11] Build a trainable Lightning-Module](https://github.com/SraRod/iThome2022/issues/21)
- [[Day 12] Training Log and History](https://github.com/SraRod/iThome2022/issues/23)

### 效能優化：資料面

- [[Day 13] 資料預處理機制與優化 - 硬體篇](https://github.com/SraRod/iThome2022/issues/25)
- [[Day 14] 資料預處理機制與優化 - 軟體篇 - 優化預處理](https://github.com/SraRod/iThome2022/issues/26)
- [[Day 15] 資料預處理機制與優化 - 軟體篇 - 空間換取時間](https://github.com/SraRod/iThome2022/issues/28)
- [[Day 16] 資料預處理機制與優化 - 軟體篇 - 土法煉鋼](https://github.com/SraRod/iThome2022/issues/31)
- [[Day 17] GPU optimization](https://github.com/SraRod/iThome2022/issues/30)

### 效能優化：模型面

- [[Day 18] Optimizer and Learning Rate](https://github.com/SraRod/iThome2022/issues/34)
- [[Day 19] Learning Rate Finder](https://github.com/SraRod/iThome2022/issues/35)
- [[Day 20] 實際訓練與結果分析](https://github.com/SraRod/iThome2022/issues/38)
- [[Day 21] Evaluation on Test Set](https://github.com/SraRod/iThome2022/issues/37)
- [[Day 22] Transfer Learning](https://github.com/SraRod/iThome2022/issues/40)
- [[Day 23] Learning Rate Warm Up](https://github.com/SraRod/iThome2022/issues/42)
- [[Day 24] Regularization in Deep Learning](https://github.com/SraRod/iThome2022/issues/44)
- [[Day 25] Label Smooth](https://github.com/SraRod/iThome2022/issues/45)
- [[Day 26] Data Augmentation](https://github.com/SraRod/iThome2022/issues/47)
- [[Day 27] Weight Decay Regularization](https://github.com/SraRod/iThome2022/issues/49)
- [[Day 28] Lp Regularization](https://github.com/SraRod/iThome2022/issues/51)

### 模型部署與結語

- [[Day 29] Model Serving](https://github.com/SraRod/iThome2022/issues/53)
- [[Day 30] 結語：回歸初心，資料是一切的根本](https://github.com/SraRod/iThome2022/issues/55)
