import torch
import torch.nn as nn

# 创建一个简单的一维信号四分类模型
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(30*32, 4),
    nn.Softmax(dim=1)
)

# 创建一个输入张量（假设为一维信号），大小为 [30, 2]
input_tensor = torch.randn(1, 30, 2)

# 将输入张量传递给模型进行前向计算
output = model(input_tensor)

print(output.shape)  # 输出预测的形状，应为 [30, 4]
