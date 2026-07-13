import torch
from torch import nn
from torch.nn import Conv2d, Linear
import torch.optim as optim
import matplotlib.pyplot as plt # 👈 1. 导入画图工具（如果报错，请在终端执行 pip install matplotlib）

class XiaoQ_cifar10(nn.Module):

    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model1(x)

# 实例化网络和损失函数
xiaoq_net = XiaoQ_cifar10()
loss = nn.CrossEntropyLoss()


# 👈 创建一个“擦拭并修改参数的手”（优化器）
# 我们把 xiaoq_net 所有的权重参数传给它，并设置学习率 lr=0.01（每次修改迈多大步）
optimizer = optim.SGD(xiaoq_net.parameters(), lr=0.01)

# 模拟真实标签：生成 64 个介于 0 到 9 之间的随机整数，代表这 64 张图各自真正的类别
# 模拟输入和真实标签
inputs = torch.ones((64, 3, 32, 32))
targets = torch.randint(0, 10, (64,), dtype=torch.long)

# ================= 🚀 记账本初始化 =================
loss_history = [] # 创建一个空列表，用来记录每一步的 Loss 轨迹
# =================================================

# ================= 真正的训练循环核心 3 步 =================

print("🚀 开始多轮循环训练...")
print("---------------------------------------")

for epoch in range(20):
    # 🚀 步骤一：把之前的旧检讨擦干净（梯度清零）
    # 必须做这一步，否则每一次算出来的修改建议会和上一次的累加在一起，导致动作变形
    optimizer.zero_grad()

    # 步骤二：前向传播（做题猜答案）
    outputs = xiaoq_net(inputs)
    result_loss = loss(outputs, targets)

    # 🚀 步骤三：反向传播（教练在线纠错，倒带计算每个参数的修改建议——梯度）
    result_loss.backward()  # 👈 就是这一句！PyTorch 会自动在后台完成全网络的链式求导！

    # 🚀 步骤四：优化器根据修改建议，真正动手调整网络参数
    optimizer.step()
    # ================= 🚀 精准记账 =================
    # 牢记：必须用 .item() 拿纯数字，否则会发生显存/内存泄漏！
    loss_history.append(result_loss.item())
    # =================================================
    print(f"Epoch [{epoch + 1:02d}/20] 完成 | 当前 Loss: {result_loss.item():.4f}")

print("\n🎉 训练完成！账本已经记满，开始绘制 Loss 进化曲线图...")

# ================= 📊 绘制曲线图 =================
plt.figure(figsize=(10, 5))          # 创建一个 10x5 英寸的画布
plt.plot(loss_history, label='Train Loss', color='blue', linewidth=2) # 画线
plt.title('XiaoQNet Training Loss Overtimes') # 给图起个标题
plt.xlabel('Epochs')                 # 横坐标是练习轮数
plt.ylabel('Loss Value')             # 纵坐标是误差值
plt.grid(True)                       # 显示网格线，方便数格子
plt.legend()                         # 显示右上角的图例说明
plt.show()                           # 弹窗显示大作！
# =================================================


