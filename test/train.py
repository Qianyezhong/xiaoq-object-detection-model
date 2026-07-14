import torch.optim
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

# 1. 依然用你写对的绝对路径
MY_PATH = r'D:\PythonProject\xiaoq-object-detection-model\test\data'

# 2. 核心调整：临时改成 download=True，逼 PyTorch 自己在本地完成解压大闭环
train_data = torchvision.datasets.CIFAR10(
    root=MY_PATH,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False  # 👈 听我的，先改成 True！
)

test_data = torchvision.datasets.CIFAR10(
    root=MY_PATH,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False  # 👈 先改成 True！
)

print(f"训练数据集的长度为:{len(train_data)}")
print(f"测试数据集的长度为:{len(test_data)}")

# 利用DataLoader进行加载数据集
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 实例化网络模型和损失函数
xiaoQnet = XiaoQNet()
loss_fn = nn.CrossEntropyLoss()

# 👈 创建一个“擦拭并修改参数的手”（优化器）
# 我们把 xiaoQnet 所有的权重参数传给它，并设置学习率 lr=0.01（每次修改迈多大步）
# 1e-3 = 1*(10)^(-3) = 1/1000 = 0.001
learning_rate = 1e-3
optimizer = torch.optim.SGD(xiaoQnet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epochs = 10

# 添加tensorboard
writer = SummaryWriter(r"D:\PythonProject\xiaoq-object-detection-model\logs_train")

print("🚀 开始多轮循环训练...")
print("---------------------------------------")
for epoch in range(epochs):
    print("------第{}轮训练开始------".format(epoch+1))
    # 训练步骤开始
    for data in train_data_loader:
        imgs, targets = data
        # 步骤一：把之前的旧检讨擦干净（梯度清零）
        optimizer.zero_grad()
        # 步骤二：前向传播（做题猜答案）
        outputs = xiaoQnet(imgs)
        loss = loss_fn(outputs, targets)
        # 步骤三：反向传播（教练在线纠错，倒带计算每个参数的修改建议——梯度）
        loss.backward()
        # 步骤四：优化器根据修改建议，真正动手调整网络参数
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 不进行优化,测试步骤开始
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            outputs = xiaoQnet(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    print("整体测试集上的Loss: {}".format(total_test_loss))
    # 准确率 = 猜对的总图片数 / 测试集总图片数 (10000张)
    epoch_accuracy = total_accuracy / len(test_data)

    total_test_step += 1
    # 💡 绝招：除以 len(test_data_loader)，把它还原成标准的平均 Loss，数字就会落在 0~2.5 之间，极其直观
    average_test_loss = total_test_loss / len(test_data_loader)
    writer.add_scalar("test_loss", average_test_loss, total_test_step)

    # 👈 新增：把准确率也同步记录到 TensorBoard 里，方便看曲线！
    writer.add_scalar("test_accuracy", epoch_accuracy, total_test_step)

    # 保存每轮模型
    torch.save(xiaoQnet, "xiaoQnet_{}.pth".format(epoch))
    print("模型保存...")

writer.close()