import torch
import torch.nn.functional as F

pred = torch.randn(4, 10)  # 预测张量，shape为[4, 10]
target = torch.randint(0, 10, (4,))  # 真实标签张量，shape为[4]
loss = F.cross_entropy(pred, target)  # 计算交叉熵损失

print(loss)