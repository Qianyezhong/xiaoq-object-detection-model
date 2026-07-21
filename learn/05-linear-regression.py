import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    # 构造Pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels), batch_size)

# print(next(iter(data_iter))) ==> X,y

# 定义模型
net = nn.Sequential(
    nn.Linear(2, 1)
)

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {l:f}')


