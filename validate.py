import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from typing import Optional, Callable
from ResNet import resnet18
from train import MMWaveDataset, CONFIG

# 修复字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False

def load_test_model(model_path, device):
    """加载训练好的模型"""
    model = resnet18(num_classes=55)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

class ToFloatTensor:
    """替代Lambda的可序列化转换类"""
    def __call__(self, x):
        return x.float()

def create_test_transform():
    """创建可序列化的测试预处理流程"""
    return transforms.Compose([
        ToFloatTensor(),
        transforms.Normalize(
            mean=CONFIG["normalize_mean"],
            std=CONFIG["normalize_std"]
        )
    ])

class SafeMMWaveDataset(MMWaveDataset):
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        super().__init__(data_dir, transform)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        try:
            data, label = super().__getitem__(idx)
            if label == -1 or data.shape != (17, *CONFIG["input_size"]):
                raise ValueError("Invalid data")
            return data, label
        except Exception as e:
            print(f"加载错误: {self.file_list[idx]} - {str(e)}")
            return torch.zeros(17, *CONFIG["input_size"]), -1

def evaluate_model(model, test_loader, device, class_names=None):
    """增强错误处理的评估函数"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 过滤无效样本
                valid_mask = labels != -1
                if not valid_mask.any():
                    continue

                inputs = inputs[valid_mask]
                labels = labels[valid_mask]

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                continue

    # 生成报告
    if not all_labels:
        raise ValueError("没有有效样本可用于验证")

    print("\n=== 分类报告 ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 简化混淆矩阵显示
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 修复多进程问题
    torch.multiprocessing.set_start_method('spawn', force=True)

    TEST_DATA_PATH = r"D:\mmWave"
    MODEL_PATH = "./checkpoints/best_model.pth"
    CLASS_NAMES = [f"Class_{i+1}" for i in range(55)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_test_model(MODEL_PATH, device)
        test_transform = create_test_transform()

        # 创建数据集
        test_dataset = SafeMMWaveDataset(TEST_DATA_PATH, transform=test_transform)

        # 过滤无效样本
        valid_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] != -1]
        test_dataset = Subset(test_dataset, valid_indices)

        # 调整数据加载器参数
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=0  # 暂时禁用多进程调试
        )

        evaluate_model(model, test_loader, device, CLASS_NAMES)

    except Exception as e:
        print(f"验证失败: {str(e)}")
        print("排查建议:")
        print("1. 检查模型文件路径是否正确")
        print("2. 验证输入数据维度是否为(17, 256, 128)")
        print("3. 确认CUDA是否可用")
        exit(1)