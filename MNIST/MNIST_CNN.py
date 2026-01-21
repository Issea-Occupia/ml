import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# 1) 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 2) 数据：MNIST (1x28x28), normalize到 [-1, 1] 左右
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
isthree = 0

train_ds = FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# 3) 模型：两层卷积 + 池化 + 全连接
"""class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 14x14 -> 14x14 (after pool)
        self.pool = nn.MaxPool2d(2, 2)                           # /2

        # 尺寸：输入 28x28
        # conv1 -> 28x28, pool -> 14x14
        # conv2 -> 14x14, pool -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,16,14,14]
        x = self.pool(F.relu(self.conv2(x)))  # [B,32,7,7]
        x = x.view(x.size(0), -1)             # [B,32*7*7]
        x = F.relu(self.fc1(x))               # [B,128]
        x = self.fc2(x)                       # logits [B,10]
        return x

model = SimpleCNN().to(device)"""
USE_3_LAYER = True  # True 就用三层，False 用两层

class SimpleCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(32*7*7, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) 
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = (SimpleCNN3() if USE_3_LAYER else SimpleCNN2()).to(device)


# 4) 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 5) 训练/测试
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    print(f"Epoch {epoch} | train loss {total_loss/total:.4f} | train acc {correct/total:.4f}")

@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    print(f"          test loss {total_loss/total:.4f} | test acc {correct/total:.4f}")

# 6) 跑起来：一般 3~5 个 epoch 就能到 98-99%+

for epoch in range(1, 10):
    train_one_epoch(epoch)
    evaluate()
