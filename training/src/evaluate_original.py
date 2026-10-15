"""
此脚本用于评估原始微调的模型（不带BiLSTM头），直接使用模型自带的分类头。
适用于评估用 ssl_model_finetuning_2.py 训练的模型检查点。

使用示例:
    python evaluate_original.py ^
    --model ./results_voc_hubert_ls_pt_data_1_6/checkpoint-1000 ^
    --df_data ./test_data.csv ^
    --output_file evaluation_results.txt ^
    --batch 32
"""

import torch
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import argparse
import numpy as np
import os

from dataset import Dataset
from utils import compute_metrics

import warnings
warnings.filterwarnings("ignore")

"""
    定义命令行参数解析器
"""
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="Evaluate an original fine-tuned model")
    parser.add_argument("--model", help="Path to the trained model checkpoint", required=True, type=str)
    parser.add_argument("--df_data", help="Path to the test data CSV with full audio paths", required=True, type=str)
    parser.add_argument("--output_file", help="Path to save the evaluation results", default="evaluation_results.txt", type=str)
    parser.add_argument("--batch", help="Evaluation batch size", default=32, type=int)
    parser.add_argument("--max_duration", help="Maximum audio duration to process", default=10.0, type=float)
    args = parser.parse_args()
    return args

"""
    主程序
"""
if __name__ == '__main__':
    args = parse_cmd_line_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("------------------------------------")
    print("Running evaluation with the following parameters:")
    print(f"  Model Checkpoint: {args.model}")
    print(f"  Test Data: {args.df_data}")
    print(f"  Batch size: {args.batch}")
    print("------------------------------------\n")

    # 1. 加载模型和特征提取器
    print(f"Loading model and feature extractor from: {args.model}")
    
    # 加载特征提取器
    try:
        print("Attempting to load feature extractor from local checkpoint...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
        print("✓ Feature extractor loaded successfully from local files.")
    except Exception as e:
        print(f"✗ Failed to load feature extractor from local files: {e}")
        print("→ Falling back to downloading from Hugging Face Hub...")
        
        # 根据模型路径推断基础模型名称
        model_name_for_fallback = "facebook/wav2vec2-base"
        if "hubert" in args.model.lower():
            model_name_for_fallback = "alkiskoudounas/voc2vec-hubert-ls-pt"
        elif "wav2vec2" in args.model.lower():
            model_name_for_fallback = "facebook/wav2vec2-base"
        
        print(f"Trying to download feature extractor for '{model_name_for_fallback}'...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_for_fallback)
        print(f"✓ Feature extractor for '{model_name_for_fallback}' downloaded successfully.")

    # 加载模型（使用标准的 AutoModelForAudioClassification）
    print("Loading model with original classification head...")
    model = AutoModelForAudioClassification.from_pretrained(
        args.model,
        local_files_only=True
    ).to(device)
    print("✓ Model loaded successfully.\n")

    # 2. 加载和准备测试数据
    print(f"Loading test data from: {args.df_data}")
    df_test = pd.read_csv(args.df_data, index_col=None)
    
    # 从模型的配置中获取label mapping
    label2id = model.config.label2id
    
    # 创建一个大小写不敏感的映射
    label2id_lower = {k.lower(): v for k, v in label2id.items()}
    
    # 转换测试集标签
    for index in range(len(df_test)):
        emotion = df_test.loc[index, "emotion"]
        # 尝试直接匹配，如果失败则尝试小写匹配
        if emotion in label2id:
            df_test.loc[index, 'label'] = label2id[emotion]
        elif emotion.lower() in label2id_lower:
            df_test.loc[index, 'label'] = label2id_lower[emotion.lower()]
        else:
            raise ValueError(f"Label '{emotion}' not found in model's label2id mapping. Available labels: {list(label2id.keys())}")
    
    df_test['label'] = df_test['label'].astype(int)
    print(f"✓ Test data loaded. Number of samples: {len(df_test)}")
    print(f"✓ Labels in model: {list(label2id.keys())}\n")

    # 3. 创建测试数据集
    test_dataset = Dataset(df_test, None, feature_extractor, args.max_duration)

    # 4. 设置Trainer用于评估
    temp_output_dir = os.path.join(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", "temp_eval_output")
    os.makedirs(temp_output_dir, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=temp_output_dir,
        per_device_eval_batch_size=args.batch,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. 运行评估
    print("Starting evaluation...")
    predictions = trainer.predict(test_dataset)
    print("✓ Evaluation finished.\n")

    # 6. 显示和保存结果
    print("=" * 60)
    print("Evaluation Metrics:")
    print("=" * 60)
    print(f"  UAR:      {predictions.metrics['test_UAR']:.4f}")
    print(f"  Accuracy: {predictions.metrics['test_accuracy']:.4f}")
    print(f"  F1 Macro: {predictions.metrics['test_f1_macro']:.4f}")
    print("=" * 60)

    with open(args.output_file, "a+") as f:
        f.write(f"Evaluated Model: {args.model}\n")
        f.write(f"Test Dataset: {args.df_data}\n")
        f.write(f"Number of Test Samples: {len(df_test)}\n")
        f.write(f"UAR: {predictions.metrics['test_UAR']:.4f}\n")
        f.write(f"Accuracy: {predictions.metrics['test_accuracy']:.4f}\n")
        f.write(f"F1 Macro: {predictions.metrics['test_f1_macro']:.4f}\n")
        f.write("------------------------------------------------\n\n")
    
    print(f"\n✓ Results have been appended to {args.output_file}")

    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_output_dir)
        print(f"✓ Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {temp_output_dir}: {e}")
