import torch
from torch import nn
from torch.nn import Conv2d, Linear

class XiaoQ_cifar10(nn.Module):

    def __init__(self):
        super().__init__()
        # H_out\W_out = (H_in - kernel_size + 2 * padding)/stride + 1
        # self.conv1 = nn.Conv2d(3, 32, 5,1, 2)
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32, 32, 5,1, 2)
        # self.max_pool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(32, 64, 5,1, 2)
        # self.max_pool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64*4*4, 64)
        # self.fc2 = nn.Linear(64, 10)

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
        # x = self.conv1(x)
        # x = self.max_pool1(x)
        # x = self.conv2(x)
        # x = self.max_pool2(x)
        # x = self.conv3(x)
        # x = self.max_pool3(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return self.model1(x)

xiaoq_net = XiaoQ_cifar10()
print(xiaoq_net)
input = torch.ones((64, 3, 32, 32))
output = xiaoq_net(input)
print(output.shape)
print(output)


