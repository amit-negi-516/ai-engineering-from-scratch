import random

random.seed(42)

# Data: y = 2x + 1
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

# Random starting weights
w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.05

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b  # forward pass
        error = pred - y  # how wrong
        total_loss += error**2  # squared error
        dw += 2 * error * x  # gradient w.r.t. w
        db += 2 * error  # gradient w.r.t. b
    dw /= len(xs)  # average gradient
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw  # update weights
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")

import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

np.random.seed(42)
w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b  # forward pass on ALL data at once
    error = pred - y
    loss = np.mean(error**2)
    dw = np.mean(2 * error * x)  # gradient — no loop over data points
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
