import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


class MyData(Dataset):

    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_names = os.listdir(self.image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        # 👈 1. 必须在返回前把图片转成神经网络认识的 Tensor
        if self.transform:
            image = self.transform(image)

        # 👈 2. 因为你后面用 imgs, targets = data 接收，这里必须返回两个值！
        # 即使暂时没有真标签，我们也先返回一个假标签（比如 0）把坑占住
        target = 0
        return image, target

# 🚀 【先造枪】定义好最基础的转换为 Tensor 的流水线
# 注意：如果你的蚂蚁图片大小不一，强烈建议在这里加上 transforms.Resize((224, 224))
# 因为 DataLoader 要求一个 Batch 里的所有图片宽高必须完全一致，否则无法打包！
my_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 强行把所有蚂蚁图统一缩放到 224x224
    transforms.ToTensor()          # 变成 Tensor 格式
])

train_dataset = MyData(
    r"D:\PythonProject\xiaoq-object-detection-model\test\test_data\train\ants_image",
    transform=my_transform)

dataloader = DataLoader(train_dataset,batch_size=64)

class XiaoqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3, stride=1)

    def forward(self,x):
        x = self.conv1(x)
        return x

xiaoq = XiaoqNet()
for data in dataloader:
    imgs, targets = data
    output = xiaoq(imgs)
    print("输入 Batch 的形状:", imgs.shape)  # 会输出类似 torch.Size([64, 3, 224, 224])
    print("输出 特征图的形状:", output.shape)  # 会输出类似 torch.Size([64, 6, 222, 222])
    print("--------------------")
