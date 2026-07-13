import torch
import torch.nn as nn

inputs = torch.tensor([[2.0,1.0,0.1]], dtype=torch.float32)
# 真实标签是第0类 （整数索引）
targets = torch.tensor([0], dtype=torch.long)

loss = nn.CrossEntropyLoss()
result = loss(inputs, targets)

print(f"PyTorch 算出来的交叉熵损失：{result.item():.3f}")