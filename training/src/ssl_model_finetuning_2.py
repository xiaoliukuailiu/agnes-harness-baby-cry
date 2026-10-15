"""
    python ssl_model_finetuning_2.py \
    --model facebook/wav2vec2-base \
    --df_data data.csv \
    --output_dir results_single_run \
    --batch 16 \
    --epochs 50 \
    --warmup_steps 500 \
    --lr 1e-4 \
    --max_duration 10.0 \
    --seed 42
    --output_file results_single_run.txt
"""

import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import EarlyStoppingCallback
import pandas as pd
import argparse
import random
import numpy as np
import os
import librosa

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, compute_metrics, compute_class_weights
    

"""
    Define Command Line Parser 
"""
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="vocal")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=16, 
        type=int,
        required=False)
    parser.add_argument(
        "--epochs",
        help="number of training epochs",
        default=50, 
        type=int,
        required=False)        
    parser.add_argument(
        "--model",
        help="model to use",
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_data",
        help="path to the data df",
        default="data.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--dataset_root",
        help="path to the dataset root",
        default="vocalization_data",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results_single_run",
        type=str,
        required=False)
    parser.add_argument(
        "--output_file",
        help="path to the output file",
        default="results_single_run.txt",
        type=str,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=500,
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum audio duration",
        default=7.0,
        type=float,
        required=False) 
    parser.add_argument(
        "--seed",
        help="seed",
        default=42,
        type=int,
        required=False)
    args = parser.parse_args()
    return args


""" 
    Read and Process Data
"""
def read_data(df_train, df_val, label_name="emotion"):
    
    ## Prepare Labels
    labels = df_train[label_name].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,label_name]]
    df_train['label'] = df_train['label'].astype(int)

    ## Validation
    for index in range(0,len(df_val)):
        df_val.loc[index,'label'] = label2id[df_val.loc[index,label_name]]
    df_val['label'] = df_val['label'].astype(int)

    return df_train, df_val, num_labels, label2id, id2label, labels


""" 
    Define model and feature extractor 
"""
def define_model(
    model_checkpoint, 
    num_labels, 
    label2id, 
    id2label, 
    device="cuda"
    ):
    
    print(f"\n正在加载模型和特征提取器: {model_checkpoint}")
    print("=" * 60)
    
    # 加载特征提取器
    feature_extractor = None
    print("\n[1/2] 加载特征提取器...")
    try:
        # 优先尝试从本地加载
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint, local_files_only=True)
        print("✓ 成功从本地加载特征提取器")
    except Exception as e:
        print(f"✗ 本地加载失败: {str(e)[:100]}")
        print("→ 正在从 Hugging Face Hub 下载...")
        try:
            # 从路径中提取模型ID
            model_id = "alkiskoudounas/voc2vec-hubert-ls-pt" if "voc2vec-hubert-ls-pt" in model_checkpoint else model_checkpoint
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            print(f"✓ 成功从网络加载特征提取器: {model_id}")
        except Exception as e2:
            print(f"✗ 网络加载也失败: {str(e2)[:100]}")
            raise RuntimeError(f"无法加载特征提取器: {model_checkpoint}") from e2

    # 加载模型
    model = None
    print("\n[2/2] 加载模型...")
    try:
        # 优先尝试从本地加载
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=True
        ).to(device)
        print("✓ 成功从本地加载模型")
    except Exception as e:
        print(f"✗ 本地加载失败: {str(e)[:100]}")
        print("→ 正在从 Hugging Face Hub 下载...")
        try:
            model_id = "alkiskoudounas/voc2vec-hubert-ls-pt" if "voc2vec-hubert-ls-pt" in model_checkpoint else model_checkpoint
            model = AutoModelForAudioClassification.from_pretrained(
                model_id, 
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True
            ).to(device)
            print(f"✓ 成功从网络加载模型: {model_id}")
        except Exception as e2:
            print(f"✗ 网络加载也失败: {str(e2)[:100]}")
            raise RuntimeError(f"无法加载模型: {model_checkpoint}") from e2
    
    print("=" * 60)
    print("模型和特征提取器加载完成!\n")
    return feature_extractor, model


""" 
    Main Program 
"""
if __name__ == '__main__':

    ## Utils 
    args = parse_cmd_line_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print("------------------------------------")
    print("Running with the following parameters:")
    print("Batch size: ", args.batch)
    print("Number of epochs: ", args.epochs)
    print("Model: ", args.model)
    print("Warmup steps: ", args.warmup_steps)
    print("Learning rate: ", args.lr)
    print("Maximum audio duration: ", args.max_duration)
    print("------------------------------------\n")

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 

    ## Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## Load the data
    data = pd.read_csv(args.df_data, index_col=None)

    ## Split the data into train and validation (80/20 split)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split_point = int(len(data) * 0.8)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    df_train = data.iloc[train_indices].reset_index(drop=True)
    df_val = data.iloc[val_indices].reset_index(drop=True)
    print("Train: ", len(df_train))   
    print("Val: ", len(df_val))

    ## Prepare Labels
    df_train, df_val, num_labels, label2id, id2label, labels = read_data(
        df_train=df_train,
        df_val=df_val,
        label_name="emotion"
        )
    print("Num labels: ", num_labels)

    ## Load the model and feature extractor
    feature_extractor, model = define_model(
        model_checkpoint=args.model, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        device=device
        )

    ## Train & Val Datasets 
    max_duration = args.max_duration
    dataset_root = args.dataset_root if args.dataset_root != "None" else None
    train_dataset = Dataset(df_train, dataset_root, feature_extractor, max_duration)
    val_dataset = Dataset(df_val, dataset_root, feature_extractor, max_duration)

    ## Training Arguments and Class Weights
    training_arguments = define_training_args(
        output_dir=output_dir, 
        batch_size=args.batch, 
        num_epochs=args.epochs,
        lr=args.lr, 
        gradient_accumulation_steps=1,
        warmup_steps=args.warmup_steps,
        )
    class_weights = compute_class_weights(df_train)

    ## Trainer 
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
        )

    ## Train and Evaluate
    trainer.train()

    ## Evaluate
    print("Evaluating...")
    predictions = trainer.predict(val_dataset)
    
    with open(args.output_file, "a+") as f:
        f.write(f"Dataset: {args.df_data}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Output Dir: {args.output_dir}\n")
        f.write(f"UAR: {predictions.metrics['test_UAR']}\n")
        f.write(f"Accuracy: {predictions.metrics['test_accuracy']}\n")
        f.write(f"F1 Macro: {predictions.metrics['test_f1_macro']}\n")
        f.write("------------------------------------------------\n\n")
