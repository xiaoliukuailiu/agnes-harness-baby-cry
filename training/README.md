# Babycry-SSL

基于语音自监督预训练模型（SSL）的婴儿哭声情感分类项目，使用 `wav2vec2` / `voc2vec` 预训练权重在下游哭声数据上微调。

## 项目简介

对 6 类婴儿哭声进行情感分类：

| 英文标签 | 中文含义 |
|----------|----------|
| angry | 生气 |
| disturbance | 不安 |
| fright | 受惊 |
| hungry | 饥饿 |
| pain | 疼痛 |
| uncomfortable | 难受不适 |

## 数据集

- 原始数据：1000 条 wav 音频（16kHz 采样），类别严重不平衡（`hungry` 占 56.5%，`pain` 仅 3%）
- 平衡策略：混合采样（少数类过采样 + 多数类欠采样），统一到每类 195 条
- 平衡后：1170 条（6 类 × 195）
- 数据清单：`data/data1_balanced.csv`（格式：`path,emotion`，`path` 为音频绝对路径）

## 训练方法

- 模型：`facebook/wav2vec2-base`（迁移预训练权重，随机初始化分类头后微调）
- 划分：80/20 训练 / 验证
- 类别加权：`WeightedTrainer` 注入 balanced class weights，缓解类别不平衡
- 关键参数：50 epochs、lr=1e-4、warmup=500、batch=16、max_duration=10s

## 实验结果

| 指标 | 数值 |
|------|------|
| UAR（宏召回率） | **96.99%** |
| Accuracy | **97.01%** |
| F1 Macro | **96.93%** |

完整结果见 `results/results_data1_balanced.txt`。

## 快速开始

```bash
# 安装依赖
pip install torch transformers librosa pandas scikit-learn accelerate

# 微调
cd src
python ssl_model_finetuning_2.py \
  --model facebook/wav2vec2-base \
  --df_data ../data/data1_balanced.csv \
  --dataset_root None \
  --output_dir results \
  --output_file results.txt \
  --batch 16 --epochs 50 --lr 1e-4 --max_duration 10.0 --seed 42
```

## 目录结构

```
├── data/data1_balanced.csv        # 平衡后数据集清单
├── src/                            # 训练 / 评估 / 推理脚本
│   ├── dataset.py                 # 数据加载
│   ├── utils.py                   # WeightedTrainer + 训练参数 + 类别权重 + 指标
│   ├── ssl_model_finetuning_2.py  # 80/20 单次划分微调
│   ├── ssl_model_finetuning.py    # 10 折交叉验证微调
│   ├── finetune_with_bilstm.py    # BiLSTM 分类头二次微调
│   ├── evaluate_original.py       # 原始分类头评估
│   ├── evaluate_with_bilstm.py    # BiLSTM 分类头评估
│   ├── analyze_dataset.py         # 数据集分布分析
│   └── predict.py                 # 单音频推理
└── results/results_data1_balanced.txt  # 评估结果
```

## 说明

- 模型 checkpoint（约 2GB，含 optimizer 状态）和原始音频文件未包含在本仓库中，如需复现请自行训练或另行获取权重。
- 数据集清单中的 `path` 为绝对路径，训练时通过 `--dataset_root None` 直接使用。
