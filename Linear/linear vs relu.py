import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

# 改这里： "linear" / "relu" / "relu2"
MODEL = "relu2"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

class LinearMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Relu1MNIST(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class Relu2MNIST(nn.Module):
    def __init__(self, h1=512, h2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def build_model(name: str) -> nn.Module:
    if name == "linear":
        return LinearMNIST()
    if name == "relu":
        return Relu1MNIST(hidden=256)
    if name == "relu2":
        return Relu2MNIST(h1=512, h2=256)
    raise ValueError("Unknown MODEL")

model = build_model(MODEL).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

@torch.no_grad
def eval_acc():
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    acc = eval_acc()
    print(f"Epoch {epoch}/{EPOCHS} | test acc = {acc:.4f}")
