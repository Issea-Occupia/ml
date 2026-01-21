import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------------
# 1) 数据
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # FashionMNIST 单通道
])

train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 2) 模型A：普通CNN
# -------------------------
class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)  # /2
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------
# 3) 模型B：ResNet18 适配 FashionMNIST
# -------------------------
def ResNet18_FMNIST():
    m = models.resnet18(weights=None)  # 不用预训练，公平对比
    # 改第一层：3通道 -> 1通道
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 改最后一层：1000类 -> 10类
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

# -------------------------
# 4) 训练/评估
# -------------------------
@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def train_one(model_name="plain", epochs=10, lr=1e-3):
    if model_name == "plain":
        model = PlainCNN()
    elif model_name == "resnet":
        model = ResNet18_FMNIST()
    else:
        raise ValueError("model_name must be 'plain' or 'resnet'")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()

        acc = evaluate(model)
        print(f"[{model_name}] epoch {ep:02d} | loss {running/len(train_loader):.4f} | test acc {acc*100:.2f}%")

    return model

# -------------------------
# 5) 直接对比跑
# -------------------------
if __name__ == "__main__":
    print("Device:", device)
    train_one("plain",  epochs=10, lr=1e-3)
    train_one("resnet", epochs=10, lr=1e-3)
