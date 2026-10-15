
import torch
import librosa
import argparse
import json
import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def predict(model_path, audio_path, device):
    """
    使用微调后的模型对单个音频文件进行分类预测。

    Args:
        model_path (str): 微调后保存的模型检查点路径 (例如 'results/checkpoint-936')。
        audio_path (str): 需要进行分类的音频文件路径。
        device (torch.device): 运行模型的设备 (cpu 或 cuda)。
    """
    # 1. 加载与模型关联的 id2label 映射
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        print(f"错误: 在 '{model_path}' 目录下找不到 config.json 文件。")
        return

    with open(config_path) as f:
        config = json.load(f)
    id2label = config['id2label']

    # 2. 加载模型和特征提取器
    print(f"正在从 '{model_path}' 加载模型...")
    model = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
    
    # 尝试加载特征提取器，如果失败则回退到原始模型
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        print("特征提取器加载成功。")
    except OSError:
        print("无法从检查点加载特征提取器，正在从原始模型 Hugging Face Hub 加载...")
        # 从原始模型加载特征提取器
        original_model = "alkiskoudounas/voc2vec-hubert-ls-pt"
        feature_extractor = AutoFeatureExtractor.from_pretrained(original_model)
        print(f"特征提取器从 {original_model} 加载成功。")
    
    model.eval() # 将模型设置为评估模式
    print("模型加载成功。")

    # 3. 加载并预处理音频文件
    print(f"正在加载音频文件: {audio_path}")
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件 '{audio_path}' 不存在。")
        return
        
    audio_array, sr = librosa.load(audio_path, sr=16000)

    # 4. 提取特征
    inputs = feature_extractor(
        audio_array, 
        sampling_rate=feature_extractor.sampling_rate, 
        padding=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # 5. 进行预测
    with torch.no_grad():
        logits = model(**inputs).logits

    # 6. 解码预测结果
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[str(predicted_id)]

    print("-" * 30)
    print(f"预测结果: {predicted_label}")
    print("-" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用微调后的 voc2vec 模型进行哭声分类。")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="微调后保存的模型检查点路径 (例如: ../../results/checkpoint-936)。"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="需要进行分类的哭声WAV文件路径。"
    )
    args = parser.parse_args()

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    predict(args.model_path, args.audio_path, device)

"""
使用示例:

1. 确保你已经完成了 `ssl_model_finetuning.py` 的训练，并且在 `results` 目录下生成了 checkpoint 文件夹。

2. 打开终端，进入 `src` 目录:
  

3. 运行以下命令:
   python predict.py --model_path ../../results/checkpoint-936 --audio_path /path/to/your/cry_sound.wav

   请将 `--model_path` 后面的路径替换成你最好的模型检查点路径。
   请将 `--audio_path` 后面的路径替换成你想要分类的实际哭声文件路径。
"""
