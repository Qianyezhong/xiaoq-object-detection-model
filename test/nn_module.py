import torch
from torch import nn


class XiaoQModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x + 1
        return output

XiaoQ = XiaoQModule()
x = torch.tensor(1.0)
output = XiaoQ(x)
print(output)