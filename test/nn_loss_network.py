import torch
from torch import nn
from torch.nn import Conv2d, Linear

class XiaoQ_cifar10(nn.Module):

    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model1(x)

loss = nn.CrossEntropyLoss()

xiaoq_net = XiaoQ_cifar10()
input = torch.ones((64, 3, 32, 32))
# 3. 模拟真实标签：生成 64 个介于 0 到 9 之间的随机整数，代表这 64 张图各自真正的类别
targets = torch.randint(0, 10, (64,), dtype=torch.long)
outputs = xiaoq_net(input)
# 按照 (模型预测, 真实标签) 的标准顺序计算损失
result_loss = loss(outputs, targets)

result_loss.backward()
# 6. 打印见证成果
print("--- 模型输出信息 ---")
print("输出的形状：", outputs.shape) # 完美契合 CIFAR-10 的 torch.Size([64, 10])
print("计算出来的交叉熵损失值：", f"{result_loss.item():.4f}")


