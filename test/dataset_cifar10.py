import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]
)

# CIFAR10数据集下载
train_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=True
)

# 用 DataLoader 打包成 Batch（成批喂给模型）
train_loader = DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=True,
)

images, labels = next(iter(train_loader))
print("一个 Batch 的图片张量形状:", images.shape) # 输出 torch.Size([64, 3, 32, 32])
print("对应的标签张量形状:", labels.shape)       # 输出 torch.Size([64])


