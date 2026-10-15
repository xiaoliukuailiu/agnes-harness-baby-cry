"""
此脚本用于在已经微调好的模型基础上添加BiLSTM分类头并进行二次微调。
它会加载一个预训练/微调过的模型，替换其分类头为BiLSTM+线性层，然后在数据集上训练。

使用示例:
    python finetune_with_bilstm.py ^
    --model D:\cryssl\results_voc_hubert_ls_pt_data_1_6\checkpoint-1000 ^
    --df_data D:\cryssl\datalabels\allcry_top4_organized.csv ^
    --dataset_root D:\cryssl\datalabels ^
    --output_dir results_bilstm_data1_6 ^
    --output_file results_bilstm.txt ^
    --batch 16 ^
    --epochs 30 ^
    --lr 5e-5 ^
    --warmup_steps 200 ^
    --seed 42
"""

import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    Wav2Vec2Config,
    HubertPreTrainedModel,
    HubertModel,
    HubertConfig,
    AutoConfig,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
import argparse
import random
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, compute_metrics, compute_class_weights

# --- 自定义模型定义 (Wav2Vec2 版本) ---
class Wav2Vec2ForClassificationWithBiLSTM(Wav2Vec2PreTrainedModel):
    """
    在Wav2Vec2基础上添加BiLSTM分类头
    """
    def __init__(self, config, bilstm_hidden_size=128, bilstm_num_layers=2):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if bilstm_num_layers > 1 else 0.0
        )
        
        # 分类器
        self.classifier = nn.Linear(bilstm_hidden_size * 2, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        lstm_output, _ = self.bilstm(hidden_states)
        pooled_output = torch.mean(lstm_output, dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# --- 自定义模型定义 (HuBERT 版本) ---
class HubertForClassificationWithBiLSTM(HubertPreTrainedModel):
    """
    在HuBERT基础上添加BiLSTM分类头
    """
    def __init__(self, config, bilstm_hidden_size=128, bilstm_num_layers=2):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hubert = HubertModel(config)
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if bilstm_num_layers > 1 else 0.0
        )
        
        # 分类器
        self.classifier = nn.Linear(bilstm_hidden_size * 2, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        lstm_output, _ = self.bilstm(hidden_states)
        pooled_output = torch.mean(lstm_output, dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

"""
    命令行参数解析
"""
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="Finetune model with BiLSTM head")
    parser.add_argument("--batch", help="batch size", default=16, type=int)
    parser.add_argument("--epochs", help="number of training epochs", default=30, type=int)
    parser.add_argument("--model", help="path to the pretrained/fine-tuned model checkpoint", required=True, type=str)
    parser.add_argument("--df_data", help="path to the data csv", required=True, type=str)
    parser.add_argument("--dataset_root", help="path to the dataset root", default=None, type=str)
    parser.add_argument("--output_dir", help="path to the output directory", required=True, type=str)
    parser.add_argument("--output_file", help="path to the output file", default="results_bilstm.txt", type=str)
    parser.add_argument("--warmup_steps", help="number of warmup steps", default=200, type=int)
    parser.add_argument("--lr", help="learning rate", default=5e-5, type=float)
    parser.add_argument("--max_duration", help="Maximum audio duration", default=10.0, type=float)
    parser.add_argument("--seed", help="random seed", default=42, type=int)
    parser.add_argument("--bilstm_hidden", help="BiLSTM hidden size", default=128, type=int)
    parser.add_argument("--bilstm_layers", help="BiLSTM number of layers", default=2, type=int)
    args = parser.parse_args()
    return args

"""
    读取和处理数据
"""
def read_data(df_train, df_val, label_name="emotion"):
    labels = df_train[label_name].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(sorted(labels)):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    for index in range(len(df_train)):
        df_train.loc[index, 'label'] = label2id[df_train.loc[index, label_name]]
    df_train['label'] = df_train['label'].astype(int)

    for index in range(len(df_val)):
        df_val.loc[index, 'label'] = label2id[df_val.loc[index, label_name]]
    df_val['label'] = df_val['label'].astype(int)

    return df_train, df_val, num_labels, label2id, id2label, labels

"""
    定义模型和特征提取器
"""
def define_model(model_checkpoint, num_labels, label2id, id2label, bilstm_hidden, bilstm_layers, device="cuda"):
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_checkpoint}")
    print(f"{'='*60}")
    
    # 1. 加载特征提取器
    print("\n[1/2] Loading feature extractor...")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint, local_files_only=True)
        print("✓ Loaded from local checkpoint")
    except Exception as e:
        print(f"✗ Local loading failed, downloading from hub...")
        model_name_fallback = "facebook/wav2vec2-base"
        if "hubert" in model_checkpoint.lower():
            model_name_fallback = "alkiskoudounas/voc2vec-hubert-ls-pt"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_fallback)
        print(f"✓ Downloaded from {model_name_fallback}")

    # 2. 加载配置并更新标签
    print("\n[2/2] Loading model with BiLSTM head...")
    config = AutoConfig.from_pretrained(model_checkpoint)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label
    
    # 3. 根据配置类型选择模型类
    if isinstance(config, HubertConfig):
        print("→ Detected HuBERT model")
        model = HubertForClassificationWithBiLSTM.from_pretrained(
            model_checkpoint,
            config=config,
            bilstm_hidden_size=bilstm_hidden,
            bilstm_num_layers=bilstm_layers,
            ignore_mismatched_sizes=True,
            local_files_only=True
        ).to(device)
    elif isinstance(config, Wav2Vec2Config):
        print("→ Detected Wav2Vec2 model")
        model = Wav2Vec2ForClassificationWithBiLSTM.from_pretrained(
            model_checkpoint,
            config=config,
            bilstm_hidden_size=bilstm_hidden,
            bilstm_num_layers=bilstm_layers,
            ignore_mismatched_sizes=True,
            local_files_only=True
        ).to(device)
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")
    
    print(f"✓ Model loaded with BiLSTM (hidden={bilstm_hidden}, layers={bilstm_layers})")
    print(f"{'='*60}\n")
    
    return feature_extractor, model

"""
    主程序
"""
if __name__ == '__main__':
    args = parse_cmd_line_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("BiLSTM Fine-tuning Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model checkpoint: {args.model}")
    print(f"Data: {args.df_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"BiLSTM hidden size: {args.bilstm_hidden}")
    print(f"BiLSTM layers: {args.bilstm_layers}")
    print(f"Max duration: {args.max_duration}s")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")

    # 创建输出目录
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"✓ Created output directory: {args.output_dir}\n")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 加载数据
    print("Loading and splitting data...")
    data = pd.read_csv(args.df_data, index_col=None)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split_point = int(len(data) * 0.8)
    train_indices, val_indices = indices[:split_point], indices[split_point:]
    
    df_train = data.iloc[train_indices].reset_index(drop=True)
    df_val = data.iloc[val_indices].reset_index(drop=True)
    print(f"✓ Train samples: {len(df_train)}")
    print(f"✓ Validation samples: {len(df_val)}\n")

    # 准备标签
    df_train, df_val, num_labels, label2id, id2label, labels = read_data(df_train, df_val, "emotion")
    print(f"Number of classes: {num_labels}")
    print(f"Class labels: {sorted(labels)}\n")

    # 加载模型和特征提取器
    feature_extractor, model = define_model(
        model_checkpoint=args.model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        bilstm_hidden=args.bilstm_hidden,
        bilstm_layers=args.bilstm_layers,
        device=device
    )

    # 创建数据集
    print("Creating datasets...")
    train_dataset = Dataset(df_train, args.dataset_root, feature_extractor, args.max_duration)
    val_dataset = Dataset(df_val, args.dataset_root, feature_extractor, args.max_duration)
    print(f"✓ Datasets created\n")

    # 训练参数和类别权重
    print("Setting up training...")
    training_arguments = define_training_args(
        output_dir=args.output_dir,
        batch_size=args.batch,
        num_epochs=args.epochs,
        lr=args.lr,
        gradient_accumulation_steps=1,
        warmup_steps=args.warmup_steps,
    )
    class_weights = compute_class_weights(df_train)
    print(f"✓ Class weights computed: {class_weights}\n")

    # 训练器
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 开始训练
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    trainer.train()

    # 评估
    print("\n" + "="*60)
    print("Evaluating on validation set...")
    print("="*60 + "\n")
    predictions = trainer.predict(val_dataset)
    
    # 显示结果
    print("="*60)
    print("Final Results:")
    print("="*60)
    print(f"UAR:      {predictions.metrics['test_UAR']:.4f}")
    print(f"Accuracy: {predictions.metrics['test_accuracy']:.4f}")
    print(f"F1 Macro: {predictions.metrics['test_f1_macro']:.4f}")
    print("="*60 + "\n")
    
    # 保存结果
    with open(args.output_file, "a+") as f:
        f.write(f"Dataset: {args.df_data}\n")
        f.write(f"Base Model: {args.model}\n")
        f.write(f"Output Dir: {args.output_dir}\n")
        f.write(f"BiLSTM Config: hidden={args.bilstm_hidden}, layers={args.bilstm_layers}\n")
        f.write(f"UAR: {predictions.metrics['test_UAR']:.4f}\n")
        f.write(f"Accuracy: {predictions.metrics['test_accuracy']:.4f}\n")
        f.write(f"F1 Macro: {predictions.metrics['test_f1_macro']:.4f}\n")
        f.write("------------------------------------------------\n\n")
    
    print(f"✓ Results saved to {args.output_file}")
    print(f"✓ Model checkpoints saved to {args.output_dir}")
    print("\nTraining complete!")
