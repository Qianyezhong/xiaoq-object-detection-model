from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, image_folder, label_folder):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_names = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return image

if __name__ == '__main__':
    train_dataset = MyData(r"D:\PythonProject\xiaoq-object-detection-model\test\Hymenoptera_data\train\ants", None)
    print(train_dataset[0])
    print(len(train_dataset))