import torch

print('1.标量与变量')
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y, x / y, x ** y)

x = torch.arange(4)
print('2.向量')
print('x:', x)
print('x[3]:', x[3])  # 通过张量的索引来访问任一元素
print('张量的形状:', x.shape)  # 张量的形状
print('张量的长度:', len(x))  # 张量的长度
z = torch.arange(24).reshape(2, 3, 4)
print('三维张量的长度:', len(z))