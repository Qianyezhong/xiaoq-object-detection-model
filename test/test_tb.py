from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")

image_folder = r"D:\PythonProject\xiaoq-object-detection-model\test\test_data\train\bees_image\16838648_415acd9e3f.jpg"
image_PIL = Image.open(image_folder)
image_array = np.array(image_PIL)

writer.add_image("test", image_array, 1, dataformats="HWC")

img = Image.open(image_folder)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image(
    "Tensor_img",
    tensor_img
)
# # y = x
# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)
writer.close()