import torch
import torchvision.models as models
import torch.nn as nn

# 🚀 召唤带有 ImageNet 预训练权重的 VGG-16
# 以前版本的写法是 pretrained=True，新版本推荐使用 weights 参数
vgg16_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# 打印一下看看，你会发现它由 features（卷积部分）和 classifier（全连接部分）优雅组合而成
print(vgg16_model)

# 模拟一张标准 ImageNet 图像送进去
mock_input = torch.randn(1, 3, 224, 224)
output = vgg16_model(mock_input)
print("最终输出形状：", output.shape) # torch.Size([1, 1000]) -> 对应 ImageNet 的 1000 个分类

vgg16_model.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_model)