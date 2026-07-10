import torch
from torch import nn

input = torch.tensor(
    [[1, 2, 0, 3, 1],
     [0, 1, 2, 3,1],
     [1, 2, 1, 0,0],
     [5, 2, 3, 1,1],
     [2, 1, 0, 1,1]]
)

input = torch.reshape(input, (-1, 1, 5, 5))

class XiaoQNet(nn.Module):
    def __init__(self):
        super(XiaoQNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        return self.maxpool(x)

xiaoq = XiaoQNet()
output = xiaoq(input)
print(output)