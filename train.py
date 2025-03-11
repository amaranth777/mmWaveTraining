import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import multiprocessing
from typing import Optional, Callable
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from torchvision import transforms

from ResNet import resnet18

# 配置参数
CONFIG = {
    "data_path": r"D:\mmWave",
    "batch_size": 32,
    "num_workers": 4,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "val_ratio": 0.2,
    "save_dir": "./checkpoints",
    "input_size": (256, 128),
    "normalize_mean": [0.5] * 17,
    "normalize_std": [0.2] * 17
}

# 自定义多通道数据增强类
class MultiChannelRandomCrop:
    """多通道安全随机裁剪"""
    def __init__(self, size):
        self.size = size  # (target_height, target_width)

    def __call__(self, tensor):
        _, h, w = tensor.shape

        # 自动调整裁剪尺寸不超过原图尺寸
        safe_height = min(h, self.size[0])
        safe_width = min(w, self.size[1])

        # 随机位置生成
        top = torch.randint(0, h - safe_height + 1, (1,)).item()
        left = torch.randint(0, w - safe_width + 1, (1,)).item()

        # 执行裁剪
        cropped = tensor[:, top:top+safe_height, left:left+safe_width]

        # 如果原图尺寸不足，进行边缘填充
        if cropped.shape[1:] != self.size:
            pad_h = max(self.size[0] - cropped.shape[1], 0)
            pad_w = max(self.size[1] - cropped.shape[2], 0)
            cropped = F.pad(cropped, [pad_w//2, pad_w - pad_w//2,pad_h//2, pad_h - pad_h//2])
        return cropped

class MultiChannelRandomRotate:
    """多通道随机旋转"""
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, tensor):
        angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        # 确保输入是4D Tensor (C, H, W) -> 添加batch维度
        rotated = torch.stack([
            F.rotate(channel.unsqueeze(0), angle, fill=channel.min().item())
            for channel in tensor
        ])
        return rotated.squeeze()

class MMWaveDataset(Dataset):
    """毫米波雷达数据集加载器"""
    def __init__(self, data_dir: str, transform: Optional[Callable] = None) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = self._validate_files()

    def _validate_files(self) -> list:
        valid_files = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".npy") and self._is_valid_filename(fname):
                valid_files.append(fname)
        return valid_files

    def _is_valid_filename(self, fname: str) -> bool:
        parts = fname.split('_')
        if len(parts) != 3 or not parts[1].isdigit():
            return False
        label = int(parts[1])
        return 1 <= label <= 55

    def __len__(self) -> int:
        return len(self.file_list)

    def _process_data(self, data: np.ndarray) -> torch.Tensor:
        """处理数据维度问题"""
        # 处理多余维度
        if data.ndim == 4 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)

        # 检查维度顺序是否正确 (通道数, 高, 宽)
        if data.shape != (17, *CONFIG["input_size"]):
            # 调整维度顺序 (假设原始数据是 H x W x C)
            if data.shape[-1] == 17:  # 如果通道在最后
                data = data.transpose(2, 0, 1)  # 转为 C x H x W
            # 转换为Tensor并调整大小
            data = torch.from_numpy(data).float()
            data = F.resize(data, CONFIG["input_size"])
        else:
            # 直接转换为Tensor
            data = torch.from_numpy(data).float()
        return data

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fname = self.file_list[idx]
        try:
            file_path = os.path.join(self.data_dir, fname)
            raw_data = np.load(file_path)
            data = self._process_data(raw_data)
            label = int(fname.split('_')[1]) - 1  # 转换为0-based索引

            if not (0 <= label < 55):
                raise ValueError(f"无效标签 {label}")

            if self.transform:
                data = self.transform(data)

            return data, label

        except Exception as e:
            print(f"加载文件 {fname} 失败: {str(e)}")
            # 返回无效数据样本（后续会被过滤）
            return torch.zeros(17, *CONFIG["input_size"], dtype=torch.float32), -1

def create_data_loaders() -> tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    transform = transforms.Compose([
        MultiChannelRandomCrop(CONFIG["input_size"]),
        MultiChannelRandomRotate(15),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(
            mean=CONFIG["normalize_mean"],
            std=CONFIG["normalize_std"]
        )
    ])

    full_dataset = MMWaveDataset(CONFIG["data_path"], transform=transform)
    valid_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1] != -1]
    full_dataset = Subset(full_dataset, valid_indices)

    dataset_size = len(full_dataset)
    val_size = int(CONFIG["val_ratio"] * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    # 验证数据集有效性
    print(f"总样本数: {len(full_dataset)}")
    print("随机检查5个样本:")
    for i in np.random.choice(len(full_dataset), 5):
        data, label = full_dataset[i]
        assert data.shape == (17, *CONFIG["input_size"]), f"样本{i}形状错误: {data.shape}"
        assert 0 <= label < 55, f"样本{i}标签错误: {label}"

    return train_loader, val_loader

def visualize_sample(loader: DataLoader):
    inputs, labels = next(iter(loader))
    print(f"批量数据维度: {inputs.shape}")
    plt.imshow(inputs[0][0].cpu().numpy(), cmap='viridis')
    plt.title(f"样本示例 (通道 0)\n标签: {labels[0].item()}")
    plt.show()

def train_model() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=55).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    train_loader, val_loader = create_data_loaders()
    visualize_sample(train_loader)

    best_val_acc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(CONFIG["save_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}]")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc*100:.2f}%")
        print("-"*50)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method("spawn", force=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    train_model()