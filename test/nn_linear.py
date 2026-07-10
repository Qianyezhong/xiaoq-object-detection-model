import torch
import torch.nn as nn
from torchvision import transforms  # 👈 导入我们不可或缺的 Tensor 转换工具
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):

    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_names = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return image

class XiaoQNet(nn.Module):
    def __init__(self):
        super(XiaoQNet, self).__init__()
        self.linear1 = nn.Linear(3*512*512, 10)

    def forward(self, x):
        # 🚨 核心修正：在喂给线性层法官之前，必须把图像彻底拍平成一维
        # start_dim=1 代表保留 Batch 维度（即使只有 1 张图，也要保留这第 0 维）
        x = torch.flatten(x, start_dim=1)
        return self.linear1(x)

xiaoq_net = XiaoQNet()

train_dataset = MyData(r"D:\PythonProject\xiaoq-object-detection-model\test\test_data\train\ants_image")
train_image = train_dataset[0].resize((512,512))
# 3. 🚀 关键步骤：把 PIL 图片过一下流水线，变成神经网络认识的 Tensor
# 此时 img_tensor 的形状是 [3, 512, 512]，数值自动归一化到了 0~1 之间
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(train_image)
# 4. 🚀 关键步骤：给单张图片穿上一件“Batch 的外衣”
# 神经网络永远要求输入必须带 Batch 维度。单张图的形状是 [3, 512, 512]，
# 我们用 unsqueeze(0) 在最前面强行加一个维度，变成 [1, 3, 512, 512]（代表一个 Batch 里只有 1 张图）
img_tensor = img_tensor.unsqueeze(0)

output = xiaoq_net(img_tensor)
print(output)
