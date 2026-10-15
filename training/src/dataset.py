import torch
import librosa
import numpy as np
import copy
import os

""" 
    Dataset Class 
"""
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, dataset_root, feature_extractor, max_duration):
        self.examples = examples['path']
        self.labels = examples['label']
        self.dataset_root = dataset_root
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.sr = 16_000
        # 确定模型期望的输入键
        if 'whisper' in self.feature_extractor.__class__.__name__.lower():
            self.input_key = 'input_features'
        else:
            self.input_key = 'input_values'

    def __getitem__(self, idx):
        try:
            # 如果 dataset_root 为 None，直接使用 examples 中的完整路径
            if self.dataset_root is None:
                audio_file = self.examples[idx]
            else:
                audio_file = os.path.join(self.dataset_root, self.examples[idx])
            speech, _ = librosa.load(audio_file, sr=self.sr)
            inputs = self.feature_extractor(
                speech.squeeze(),
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors="pt",
                max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                truncation=True,
                padding='max_length')
            
            # 使用正确的键
            input_tensor = inputs.get('input_values')
            if input_tensor is None:
                input_tensor = inputs.get('input_features')

            item = {self.input_key: input_tensor.squeeze()}
            item["labels"] = torch.tensor(self.labels[idx])
        except Exception as e:
            print(f"处理文件时出错 {self.examples[idx]}: {e}")
            # 返回一个随机的有效样本
            random_idx = np.random.randint(len(self.examples))
            if self.dataset_root is None:
                audio_file = self.examples[random_idx]
            else:
                audio_file = os.path.join(self.dataset_root, self.examples[random_idx])
            speech, _ = librosa.load(audio_file, sr=self.sr)
            inputs = self.feature_extractor(
                speech.squeeze(),
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors="pt",
                max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                truncation=True,
                padding='max_length')
            
            input_tensor = inputs.get('input_values')
            if input_tensor is None:
                input_tensor = inputs.get('input_features')

            item = {self.input_key: input_tensor.squeeze()}
            item["labels"] = torch.tensor(self.labels[random_idx])
        return item

    def __len__(self):
        return len(self.examples)