import numpy as np
from torchvision.datasets import MNIST

# =========================
# 0) 数据准备
# =========================
dataset = MNIST(root="./data", train=True, download=True)

def img_to_x(img):
    # 统一做归一化，不然数值尺度太大（你原来训练时没 /255，会很难收敛）
    A = np.array(img).astype(np.float32) / 255.0
    return A.reshape(784)

# =========================
# 1) 数学组件：softmax / loss
# =========================
def softmax(z):
    z = z - np.max(z)          # 稳定性
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def loss_one(scores, y):
    p = softmax(scores)
    return -np.log(p[y] + 1e-12)

# =========================
# 2) 激活函数：ReLU / Sigmoid
# =========================
def relu(u):
    return np.maximum(0.0, u)

def drelu(u):
    return (u > 0).astype(np.float32)

def sigmoid(u):
    # 稳定一点的写法：避免 exp 溢出
    # 对大正数/大负数都更稳
    out = np.empty_like(u, dtype=np.float32)
    pos = (u >= 0)
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    exp_u = np.exp(u[neg])
    out[neg] = exp_u / (1.0 + exp_u)
    return out

def dsigmoid(a):
    # 如果 a = sigmoid(u)，那么导数 a*(1-a)
    return a * (1.0 - a)

# =========================
# 3) 两层网络：784 -> H -> 10
# =========================
def init_params(hidden=128, seed=0):
    rng = np.random.default_rng(seed)
    W1 = (rng.normal(0, 1.0, size=(hidden, 784)).astype(np.float32)
          * np.sqrt(2.0 / 784))
    b1 = np.zeros(hidden, dtype=np.float32)

    W2 = (rng.normal(0, 1.0, size=(10, hidden)).astype(np.float32)
          * np.sqrt(2.0 / hidden))
    b2 = np.zeros(10, dtype=np.float32)
    return W1, b1, W2, b2

def forward_two_layer(x, W1, b1, W2, b2, act="relu"):
    """
    返回：scores, cache（反传要用）
    """
    u1 = W1 @ x + b1                # [H]
    if act == "relu":
        a1 = relu(u1)               # [H]
    elif act == "sigmoid":
        a1 = sigmoid(u1)            # [H]
    else:
        raise ValueError("act must be 'relu' or 'sigmoid'")

    scores = W2 @ a1 + b2            # [10]
    cache = (x, u1, a1)              # 反传需要
    return scores, cache

def train_one_two_layer(x, y, W1, b1, W2, b2, lr=0.1, act="relu"):
    """
    单样本 SGD 更新一次
    """
    # ---- forward
    scores, cache = forward_two_layer(x, W1, b1, W2, b2, act=act)
    p = softmax(scores)
    L = -np.log(p[y] + 1e-12)

    # ---- backward: softmax + CE 的经典梯度
    ds = p.copy()
    ds[y] -= 1.0                    # [10]  = dL/dscores

    x0, u1, a1 = cache

    # layer2 grads
    dW2 = ds[:, None] * a1[None, :]  # [10,H]
    db2 = ds                          # [10]
    da1 = W2.T @ ds                   # [H]

    # activation backprop
    if act == "relu":
        du1 = da1 * drelu(u1)         # [H]
    else:  # sigmoid
        du1 = da1 * dsigmoid(a1)      # [H]

    # layer1 grads
    dW1 = du1[:, None] * x0[None, :]  # [H,784]
    db1 = du1                          # [H]

    # ---- update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return L, W1, b1, W2, b2

# =========================
# 4) 评估函数
# =========================
def accuracy_on_subset_two_layer(dataset, W1, b1, W2, b2, act="relu", n=2000):
    correct = 0
    for i in range(n):
        img, y = dataset[i]
        x = img_to_x(img)
        scores, _ = forward_two_layer(x, W1, b1, W2, b2, act=act)
        pred = int(np.argmax(scores))
        correct += (pred == y)
    return correct / n

# =========================
# 5) 训练入口：跑 ReLU vs Sigmoid
# =========================
def run_experiment(act="relu", hidden=128, lr=0.1, epochs=10, steps_per_epoch=20000, seed=0):
    W1, b1, W2, b2 = init_params(hidden=hidden, seed=seed)

    # 训练前看看准确率
    acc0 = accuracy_on_subset_two_layer(dataset, W1, b1, W2, b2, act=act, n=2000)
    print(f"[{act}] acc(before) on first 2000 = {acc0:.3f}")

    for ep in range(epochs):
        total_loss = 0.0
        for i in range(steps_per_epoch):
            img, y = dataset[i]  # 为了“最小改动”仍然用固定前 N 张；更合理是打乱抽样
            x = img_to_x(img)
            Li, W1, b1, W2, b2 = train_one_two_layer(x, y, W1, b1, W2, b2, lr=lr, act=act)
            total_loss += float(Li)

        avg_loss = total_loss / steps_per_epoch
        acc = accuracy_on_subset_two_layer(dataset, W1, b1, W2, b2, act=act, n=2000)
        print(f"[{act}] epoch {ep+1}/{epochs} | avg_loss={avg_loss:.4f} | acc(first2000)={acc:.3f}")

    return W1, b1, W2, b2

# ============ 运行对比 ============
# 建议：ReLU 用 lr=0.1 或 0.05；Sigmoid 常需要更小 lr（比如 0.05 或 0.01）
print("=== ReLU model ===")
params_relu = run_experiment(act="relu", hidden=128, lr=0.1, epochs=10, steps_per_epoch=20000, seed=0)

print("\n=== Sigmoid model ===")
params_sig = run_experiment(act="sigmoid", hidden=128, lr=0.05, epochs=10, steps_per_epoch=20000, seed=0)
