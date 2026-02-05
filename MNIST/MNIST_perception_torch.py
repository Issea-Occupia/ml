import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 0) 数据：MNIST + ToTensor() 会把像素变成 [0,1] 的 float tensor
transform = transforms.ToTensor()
trainset = MNIST(root="./data", train=True, download=True, transform=transform)
testset  = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# 1) 模型：就是一个权重矩阵 W (10x784) + 偏置 b(10)
model = nn.Linear(28 * 28, 10).to(device)

# 2) 损失：CrossEntropyLoss = (LogSoftmax + NLLLoss) 的合体（更稳定）
criterion = nn.CrossEntropyLoss()

# 3) 优化器：负责按梯度更新参数（SGD / Adam 都可）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def evaluate():
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)            # (B,1,28,28) -> (B,784)
            logits = model(x)                    # (B,10)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)

            pred = logits.argmax(dim=1)          # 取最大logit的类
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

# 4) 训练
epochs = 5
for ep in range(1, epochs + 1):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)                # (B,784)

        optimizer.zero_grad()                    # 清空上一轮梯度
        logits = model(x)                        # 前向：Wx+b
        loss = criterion(logits, y)              # loss（内部做了log-softmax）
        loss.backward()                          # 反向：自动算 dW, db
        optimizer.step()                         # 更新：W,b 往loss下降方向走一步

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = loss_sum / total
    train_acc = correct / total
    test_loss, test_acc = evaluate()
    print(f"Epoch {ep} | train loss {train_loss:.4f} acc {train_acc:.4f} "
          f"| test loss {test_loss:.4f} acc {test_acc:.4f}")
