from PIL import Image
from torchvision.transforms import transforms

img = Image.open(r"D:\PythonProject\xiaoq-object-detection-model\test\test_data\train\ants_image\0013035.jpg")

img_to_tensor = transforms.ToTensor()
img_tensor = img_to_tensor(img)

# normalize
# trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# img_norm = trans_norm(img_tensor)
# print(img_norm)

# resize
# print("Resize之前的尺寸=",img.size)
# trans_resize = transforms.Resize((512,512))
# img_resize = trans_resize(img)
# print("Resize之后的尺寸=",img_resize)

my_pipline = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
img_compose = my_pipline(img)
print(img_compose)