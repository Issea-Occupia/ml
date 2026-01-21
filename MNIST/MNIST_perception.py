import numpy as np
from torchvision.datasets import MNIST

dataset = MNIST(root="./data",train=True,download=True)

img, label = dataset[0] 
A = np.array(img)         

A = A.astype(np.float32) 

x = A.reshape(784)       

rng = np.random.default_rng(0)
W = rng.normal(0, 0.01, size=(10, 784)).astype(np.float32)
b = np.zeros(10, dtype=np.float32)

scores = W @ x + b        
pred = int(np.argmax(scores))

print(f"true label ={label}" )
print(f"pred      ={pred}" )
print(f"scores    ={scores}" )
def softmax(z):
    z = z - np.max(z)  
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

p = softmax(scores)
print(f"softmax = {p}")

def loss_one(scores, y):
    p = softmax(scores)
    return -np.log(p[y] + 1e-12)

L = loss_one(scores, label)
print("loss =", float(L))

def train_one(x, y, W, b, lr=0.1):
    scores = W @ x + b          
    p = softmax(scores)          
    L = -np.log(p[y] + 1e-12)

    ds = p.copy()
    ds[y] -= 1               
    dW = ds[:, None] * x[None, :]
    db = ds                     
    W -= lr * dW
    b -= lr * db
    return L, W, b
L0 = loss_one(W @ x + b, label)
L1, W, b = train_one(x, label, W, b, lr=0.1)
print("loss before =", float(L0))
print("loss after  =", float(L1))
def accuracy_on_subset(dataset, W, b, n=200):
    correct = 0
    for i in range(n):
        img, y = dataset[i]
        A = np.array(img).astype(np.float32) / 255.0
        x = A.reshape(784)
        pred = int(np.argmax(W @ x + b))
        if pred == y:
            correct += 1
    return correct / n
print(f"acc(before) on first 200 = {accuracy_on_subset(dataset, W, b, n=200)}")

lr = 0.01
epochs = 30
steps_per_epoch = 50000
for ep in range(epochs):
    total_loss = 0.0
    for i in range(steps_per_epoch):
        img, y = dataset[i]
        A = np.array(img).astype(np.float32) 
        x = A.reshape(784)
        Li, W, b = train_one(x, y, W, b, lr=lr)
        total_loss += float(Li)

    avg_loss = total_loss / steps_per_epoch
    acc = accuracy_on_subset(dataset, W, b, n=2000)
    print(f"epoch {ep+1}/{epochs} | avg_loss={avg_loss:.4f} | acc(first2000)={acc:.3f}")

print(f"acc(after) on first 200  = {accuracy_on_subset(dataset, W, b, n=200)}")