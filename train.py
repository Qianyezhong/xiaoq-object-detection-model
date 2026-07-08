import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import XiaoqModel
from yolo_dataset import YOLODataset
from loss import XiaoqLoss

# def collate_fn(batch):
#     images = []
#     targets = []
#     for image,target in batch:
#         images.append(image)
#         targets.append(target)
#     images = torch.stack(images)
#     return images, targets

def train():
    # 1.设备
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # 2.数据集
    dataset = YOLODataset(
        image_folder= r"D:\PythonProject\xiaoq-object-detection-model\Single_object_yolo_dataset\train\images",
        label_folder= r"D:\PythonProject\xiaoq-object-detection-model\Single_object_yolo_dataset\train\labels",
        transform= transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ]),
        label_transform= None
    )

    dataloader = DataLoader(
        dataset,
        batch_size= 4,
        shuffle= True,
        # collate_fn= collate_fn
    )

    # 3.模型
    model = XiaoqModel()
    model.to(device)

    # 4.loss Function
    criterion = XiaoqLoss()
    criterion.to(device)

    # 5.优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr= 0.0001
    )

    # 6.开始训练
    epochs = 200

    print("🚀 开始训练...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            # 向前传播
            outputs = model(images)

            # loss
            loss = criterion(
                outputs,
                targets
            )
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch}, Loss:{total_loss}"
        )
    # 💡 👈 【核心：就是在这里为你补齐了保存逻辑！】
    # 它处于 for 循环外面，代表 100 个 Epoch 全部顺利跑完后，才会执行并保存
    weight_path = "xiaoq_model.pth"
    torch.save(model.state_dict(), weight_path)
    print("\n" + "=" * 40)
    print(f"🎉 训练完美结束！权重文件已成功保存至:\n👉 {weight_path}")
    print("=" * 40)
if __name__ == '__main__':
    train()