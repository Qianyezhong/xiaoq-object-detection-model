import torch
import cv2
from PIL import Image
from torchvision import transforms

from model import XiaoqModel


def predict(image_path, model_path, device):
    # 1.定义类别映射表
    class_names = ["猫","狗","马","大象"]

    # 2.初始化模型并加载刚训练好的权重
    model = XiaoqModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # 开启评估模式

    # 3.使用PIL  读取测试的数据集
    pil_image = Image.open(image_path).convert('RGB')
    # 另外保留一个用于 OpenCV 弹窗画框的原图
    ori_image = cv2.imread(image_path)
    if ori_image is None:
        print(f"❌️ 无法正确读取图片，请检查文件路径是否正确: {image_path}")
        return
    orig_h, orig_w, _= ori_image.shape

    # 4.图像预处理（需要和train.py的Resize和Tensor转换完全一致）
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        # 👈 核心：这里漏掉了解析！必须加上和 train.py 一模一样的 Normalize
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(device) # 升维度,变成[1,3,448,448]

    # 5.模型推理
    with torch.no_grad(): # 关闭梯度计算，能够省内存且速度极快
        output = model(input_tensor) # 得到 [1,8]
        output = output.squeeze(0) # 展平成 [8]

    # 6.解析预测的位置坐标（经过sigmoid,范围在 0-1）
    pred_location = output[0:4].cpu().numpy() # NumPy 不支持 GPU Tensor,所以需要将数据从GPU转到CPU
    x_c, y_c, w, h = pred_location[0], pred_location[1], pred_location[2], pred_location[3]

    # 7.解析类别和执行度（通过softmax 把后面四位的logits变成概率）
    pred_clas_logits = output[4:8]
    pred_cls_probs = torch.softmax(pred_clas_logits, dim=0)
    class_idx = torch.argmax(pred_cls_probs).item()
    confidence = pred_cls_probs[class_idx].item()

    # 8.核心逆向转换：将YOLO相对比例坐标还原成图像上的真实像素坐标
    x1 = int((x_c - w / 2) * orig_w)
    y1 = int((y_c - h / 2) * orig_h)
    x2 = int((x_c + w / 2) * orig_w)
    y2 = int((y_c + h / 2) * orig_h)

    # 防止数组越界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(x2, orig_w), min(y2, orig_h)

    # 9.终端打印结果
    print("\n" + "=" * 30)
    print(f"🎯 【模型预测成功】")
    print(f"预测类别 : {class_names[class_idx]} (类别索引: {class_idx})")
    print(f"置信度(把握) : {confidence * 100:.2f}%")
    print(f"预测的归一化中心坐标 [x_c, y_c, w, h]:\n    [{x_c:.4f}, {y_c:.4f}, {w:.4f}, {h:.4f}]")
    print(f"还原后的像素矩形框 [左上角x1, y1, 右下角x2, y2]:\n    [{x1}, {y1}, {x2}, {y2}]")
    print("=" * 30 + "\n")

    # 10. 在图上画一个【绿色】的预测框（为了区别于原本的红框）
    label_text = f"{class_names[class_idx]} {confidence:.2f}"
    cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色线，粗细为2
    cv2.putText(ori_image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 11. 弹出窗口查看结果，并保存在本地
    cv2.imshow("Xiaoq ResNet18 Predict", ori_image)
    cv2.imwrite("predict_result.jpg", ori_image)  # 保存一张预测图到项目根目录下
    print("💡 结果图片已保存为 predict_result.jpg，在弹出窗口中按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 📝 这里改成你想测试的那张纯色红框图片的路径
    test_image_path = r"D:\PythonProject\xiaoq-object-detection-model\Single_object_yolo_dataset\val\images\21.jpg"

    # 📝 这里是你训练完 100 轮通过 torch.save 保存的权重文件路径
    model_weight_path = r"D:\PythonProject\xiaoq-object-detection-model\xiaoq_model.pth"

    # 自动切换显卡或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict(test_image_path, model_weight_path, device)