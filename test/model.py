import torch
import torch.nn as nn

class XiaoQNet(nn.Module):
    def __init__(self):
        super(XiaoQNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    xiaoQnet = XiaoQNet()
    input = torch.ones((64,3,32,32))
    output = xiaoQnet(input)
    print(output.shape)