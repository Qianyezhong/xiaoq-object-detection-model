from torchvision import transforms
from PIL import Image

# tensor数据类型
# 通过transformer.ToTensor

image_folder = r"D:\PythonProject\xiaoq-object-detection-model\test\Hymenoptera_data\train\ants\0013035.jpg"
image = Image.open(image_folder)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image)
print(tensor_img, tensor_img.shape)