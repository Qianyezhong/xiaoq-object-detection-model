import torch.onnx
from torch import nn
from torchvision.models import vgg16, resnet18, ResNet18_Weights


class XiaoqModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 采用ResNet17
        resnet = resnet18(
            weights= ResNet18_Weights.DEFAULT
        )
        # self.feature_export = vgg16().features
        # self.fc_layer = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512*14*14, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     # 最后输出是 center四个坐标+四个类别（with helmet...）
        #     nn.Linear(1024, 8)
        # )

        # 去掉最后分类层
        self.feature_export = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                512,
                256
            ),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x= self.feature_export(x)
        x= self.fc_layer(x)

        box = torch.sigmoid(
            x[:,:4]
        )
        cls = x[:,4:]
        return torch.cat([
            box,
            cls
        ], 1)

if __name__ == '__main__':
    model = XiaoqModel()
    # dataset = YOLODataset(r"D:\DeepLearning\HelmetDataset-YOLO-Train\images",
    #                       r"D:\DeepLearning\HelmetDataset-YOLO-Train\labels",
    #                       transforms.Compose([
    #                           transforms.ToTensor(),
    #                           transforms.Resize((512, 512)),
    #                       ]),
    #                       None)
    # images, target = dataset[0]
    # output = model(images)
    # print(output.shape)
    # torch.onnx.export(model, images, "xiaoq-conv2d.onnx")

    input = torch.randn(1, 3, 448, 448)
    output = model(input)
    print(output)
    print(output.shape)