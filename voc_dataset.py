import os
import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder) #获取文件列表,['1.jpg','2.jpg']

        # 类别转换
        self.classes_list = ["no helmet", "motor", "number", "with helmet"]

    def __len__(self):
        return len(self.image_names) #返回文件长度

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # new1.png -> new1.xml
        # new1.png -> [new1,png] -> new1 + ".xml"
        label_name = image_name.split('.')[0] + ".xml"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path, 'r', encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict["annotation"]["object"]
        target = []
        for obj in objects:
            object_name = obj["name"]
            # object_name -> id 比如 no helmet -> 0
            object_class_id = self.classes_list.index(object_name)
            object_xmax = float(obj["bndbox"]["xmax"])
            object_ymax = float(obj["bndbox"]["ymax"])
            object_xmin = float(obj["bndbox"]["xmin"])
            object_ymin = float(obj["bndbox"]["ymin"])
            target.append([object_class_id, object_xmin, object_ymin, object_xmax, object_ymax])
        # label 转 tensor类型
        target = torch.Tensor(target)
        # image 转 tensor类型
        if self.transform is not None:
            image = self.transform(image)
        return image, target

if __name__ == '__main__':
    train_dataset = VOCDataset(r"D:\DeepLearning\HelmetDataset-VOC\train\images",r"D:\DeepLearning\HelmetDataset-VOC\train\labels", transforms.Compose([transforms.ToTensor()]),None)
    print(len(train_dataset))
    print(train_dataset[0])