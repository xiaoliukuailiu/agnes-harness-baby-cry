"""
数据集类别分布分析脚本
用于检查训练数据中各类别的样本数量分布
"""

import pandas as pd
import argparse
from collections import Counter

def analyze_dataset(csv_path):
    """
    分析数据集中各类别的分布情况
    
    Args:
        csv_path: CSV文件路径
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 检查是否有emotion列
    if 'emotion' not in df.columns:
        print(f"错误: CSV文件中没有找到 'emotion' 列")
        print(f"可用的列: {df.columns.tolist()}")
        return
    
    # 统计各类别数量
    emotion_counts = df['emotion'].value_counts()
    total_samples = len(df)
    
    print("\n" + "="*60)
    print("数据集类别分布分析")
    print("="*60)
    print(f"\n总样本数: {total_samples}\n")
    print(f"{'类别':<15} {'数量':<10} {'占比':<10} {'可视化'}")
    print("-"*60)
    
    for emotion, count in emotion_counts.items():
        percentage = (count / total_samples) * 100
        bar = "█" * int(percentage / 2)  # 每个█代表2%
        print(f"{emotion:<15} {count:<10} {percentage:>5.1f}%    {bar}")
    
    print("-"*60)
    
    # 计算不平衡比率
    max_count = emotion_counts.max()
    min_count = emotion_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n最多类别样本数: {max_count}")
    print(f"最少类别样本数: {min_count}")
    print(f"不平衡比率: {imbalance_ratio:.2f}:1")
    
    # 评估不平衡程度
    print("\n" + "="*60)
    if imbalance_ratio < 2:
        print("✓ 数据集相对平衡")
    elif imbalance_ratio < 5:
        print("⚠ 数据集存在轻度不平衡")
        print("  建议: 当前的类别权重已经在处理，但可以考虑数据增强")
    elif imbalance_ratio < 10:
        print("⚠⚠ 数据集存在中度不平衡")
        print("  建议:")
        print("  1. 增加少数类样本（数据增强、收集更多数据）")
        print("  2. 考虑使用 Focal Loss")
        print("  3. 调整类别权重的计算方式")
    else:
        print("⚠⚠⚠ 数据集严重不平衡")
        print("  建议:")
        print("  1. 重点收集少数类样本")
        print("  2. 使用过采样技术（SMOTE等）")
        print("  3. 使用 Focal Loss 代替交叉熵")
        print("  4. 考虑使用集成方法")
    
    print("="*60 + "\n")
    
    # 建议的改进措施
    print("改进建议:")
    print("-"*60)
    print("1. 数据增强:")
    print("   - 时间拉伸/压缩")
    print("   - 音高变换")
    print("   - 添加背景噪声")
    print("   - SpecAugment")
    print()
    print("2. 调整训练策略:")
    print("   - 使用更激进的类别权重")
    print("   - 降低学习率，增加训练轮数")
    print("   - 使用 Focal Loss (gamma=2.0)")
    print()
    print("3. 评估指标:")
    print("   - 重点关注 F1-Macro 和 UAR")
    print("   - 查看混淆矩阵，了解具体哪些类别被混淆")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分析数据集类别分布")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="CSV数据集文件路径"
    )
    args = parser.parse_args()
    
    analyze_dataset(args.csv_path)

"""
使用示例:
python analyze_dataset.py --csv_path D:\cryssl\data1\data1_top4.csv
"""
