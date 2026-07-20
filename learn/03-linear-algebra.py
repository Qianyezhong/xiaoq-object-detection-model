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

print('3.矩阵')
A = torch.arange(20).reshape(5, 4)
print('A:', A)
print('A.T:', A.T)  # 矩阵的转置

# 对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B==B.T)

# 矩阵运算
print('4.矩阵的计算')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:', A)
print('B:', B)
print('A + B:', A + B)  # 矩阵相加
print('A * B:', A * B)  # 矩阵相乘

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('X:', X)
print('a + X:', a + X)  # 矩阵的值加上标量
print('a * X:', a * X)
print((a * X).shape)

# 计算元素和
C = torch.arange(4, dtype=torch.float32)
print(C.sum())
# 降维
print(A)
print('A.shape:', A.shape)
print('A.sum():', A.sum())
print('A.sum(axis=0):', A.sum(axis=0))  # 沿0轴汇总以生成输出向量(列输出；比如第一列的所有行相加输出)
print('A.sum(axis=1):', A.sum(axis=1))  # 沿1轴汇总以生成输出向量(行输出)
# 求平均
print('A.mean():', A.mean())
print('A.sum() / A.numel():', A.sum() / A.numel())

# 点积
x = torch.tensor([0.,1,2,3])
y = torch.ones(4, dtype = torch.float32)
print('x与y的点积:', torch.dot(x, y))
print('x与y的点积:', torch.sum(x * y))

# 向量积
print('7.矩阵-向量相乘(向量积)')
print('A:', A)  # 5*4维
print('x:', x)  # 4*1维
print('torch.mv(A, x):', torch.mv(A, x))

# 矩阵乘法
print('8.矩阵-矩阵相乘(向量积)')
print('A:', A)  # 5*4维
B = torch.ones(4, 3)  # 4*3维
print('B:', B)
print('torch.mm(A, B):', torch.mm(A, B))

# 范数
print('9.范数')
u = torch.tensor([3.0, -4.0])
print('向量的𝐿2范数:', torch.norm(u))  # 向量的𝐿2范数
print('向量的𝐿1范数:', torch.abs(u).sum())  # 向量的𝐿1范数