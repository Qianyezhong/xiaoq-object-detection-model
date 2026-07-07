import torch.onnx
from torch import nn
from torchvision.models import vgg16

class XiaoqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_export = vgg16().features
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            # 最后输出是 center四个坐标+四个类别（with helmet...）
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        x= self.feature_export(x)
        return self.fc_layer(x)

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