import torch
from torch import nn
import torch.nn.functional as F

class XiaoQModule(nn.Module):

    def __init__(self):
        super(XiaoQModule, self).__init__()
        self.conv2 = nn.Conv2d(1,1,3,1)

    def forward(self, x):
        return self.conv2(x)

input = torch.tensor(
    [[1., 2, 0, 3, 1],  # 👈 只要把 1 改成 1.
     [0, 1, 2, 3, 1],
     [1, 2, 1, 0, 0],
     [5, 2, 3, 1, 1],
     [2, 1, 0, 1, 1]]
)

kernel = torch.tensor(
    [[1., 2, 1],        # 👈 这里的 1 也改成 1.
     [0, 1, 0],
     [2, 1, 0]]
)

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output1 = F.conv2d(input, kernel, stride=1)

XiaoQModule = XiaoQModule()
output2 = XiaoQModule(input)

print(output1)
print(output2)