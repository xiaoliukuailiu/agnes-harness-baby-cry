"""
此脚本专门用于评估一个已经训练好的、带有BiLSTM头的音频分类模型。
它会加载一个模型检查点和一个测试数据集，然后计算并报告评估指标。

使用示例:
    python evaluate_with_bilstm.py ^
    --model ./results_bilstm_run/checkpoint-500 ^
    --df_data ./test_data.csv ^
  
    --output_file evaluation_results.txt ^
    --batch 24
"""

import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    HubertPreTrainedModel,
    HubertModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
    Wav2Vec2Config,
    HubertConfig,
)
from transformers.modeling_outputs import CausalLMOutput
import pandas as pd
import argparse
import numpy as np
import os

# 假设dataset.py和utils.py在同一个src目录下或Python路径中
from dataset import Dataset
from utils import compute_metrics

import warnings
warnings.filterwarnings("ignore")

# --- 自定义模型定义 (Wav2Vec2 版本) ---
class Wav2Vec2ForClassificationWithBiLSTM(Wav2Vec2PreTrainedModel):
    """
    这个自定义模型在Wav2Vec2的Transformer输出之上添加了一个BiLSTM层用于分类。
    """
    def __init__(self, config, bilstm_hidden_size=128, bilstm_num_layers=2):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
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
        mean_pooled_output = torch.mean(lstm_output, dim=1)
        logits = self.classifier(mean_pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

# --- 自定义模型定义 (HuBERT 版本) ---
class HubertForClassificationWithBiLSTM(HubertPreTrainedModel):
    """
    这个自定义模型在HuBERT的Transformer输出之上添加了一个BiLSTM层用于分类。
    """
    def __init__(self, config, bilstm_hidden_size=128, bilstm_num_layers=2):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
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
        mean_pooled_output = torch.mean(lstm_output, dim=1)
        logits = self.classifier(mean_pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


"""
    定义命令行参数解析器
"""
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="Evaluate a model with a BiLSTM head")
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
    
    try:
        # 优先尝试从本地检查点加载特征提取器
        print("Attempting to load feature extractor from local checkpoint...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
        print("✓ Feature extractor loaded successfully from local files.")
    except Exception as e:
        # 如果本地加载失败，则从网络回退
        print(f"✗ Failed to load feature extractor from local files: {e}")
        print("→ Falling back to downloading from Hugging Face Hub...")
        
        # 根据模型路径推断基础模型名称
        model_name_for_fallback = "facebook/wav2vec2-base" # 默认值
        if "hubert" in args.model.lower():
            model_name_for_fallback = "alkiskoudounas/voc2vec-hubert-ls-pt"
        
        print(f"Trying to download feature extractor for '{model_name_for_fallback}'...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_for_fallback)
        print(f"✓ Feature extractor for '{model_name_for_fallback}' downloaded successfully.")

    # 动态加载模型 - 根据配置类型选择正确的模型类
    config = AutoConfig.from_pretrained(args.model)
    
    if isinstance(config, HubertConfig):
        print("Detected HuBERT model. Loading with HubertForClassificationWithBiLSTM...")
        model = HubertForClassificationWithBiLSTM.from_pretrained(
            args.model,
            config=config,
            local_files_only=True
        ).to(device)
    elif isinstance(config, Wav2Vec2Config):
        print("Detected Wav2Vec2 model. Loading with Wav2Vec2ForClassificationWithBiLSTM...")
        model = Wav2Vec2ForClassificationWithBiLSTM.from_pretrained(
            args.model,
            config=config,
            local_files_only=True
        ).to(device)
    else:
        raise TypeError(f"Unsupported model config type: {type(config)}. Only HubertConfig and Wav2Vec2Config are supported.")
    
    print("✓ Model and feature extractor loaded successfully.\n")

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
    print(f"✓ Test data loaded. Number of samples: {len(df_test)}\n")

    # 3. 创建测试数据集
    # dataset_root 设置为 None，因为我们假设 CSV 文件中的路径是完整的
    test_dataset = Dataset(df_test, None, feature_extractor, args.max_duration)

    # 4. 设置一个临时的Trainer用于评估
    # 我们只需要一个输出目录来存放临时的评估文件
    temp_output_dir = os.path.join(os.path.dirname(args.output_file), "temp_eval_output")
    if not os.path.isdir(temp_output_dir):
        os.makedirs(temp_output_dir, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=temp_output_dir,
        per_device_eval_batch_size=args.batch,
        report_to=[], # 关闭wandb等报告
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

    # 6. 保存结果
    print("Evaluation Metrics:")
    print(f"  UAR: {predictions.metrics['test_UAR']:.4f}")
    print(f"  Accuracy: {predictions.metrics['test_accuracy']:.4f}")
    print(f"  F1 Macro: {predictions.metrics['test_f1_macro']:.4f}")

    with open(args.output_file, "a+") as f:
        f.write(f"Evaluated Model: {args.model}\n")
        f.write(f"Test Dataset: {args.df_data}\n")
        f.write(f"UAR: {predictions.metrics['test_UAR']:.4f}\n")
        f.write(f"Accuracy: {predictions.metrics['test_accuracy']:.4f}\n")
        f.write(f"F1 Macro: {predictions.metrics['test_f1_macro']:.4f}\n")
        f.write("------------------------------------------------\n\n")
    
    print(f"\nResults have been appended to {args.output_file}")

    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_output_dir)
        print(f"Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e:
        print(f"Could not remove temporary directory {temp_output_dir}: {e}")

