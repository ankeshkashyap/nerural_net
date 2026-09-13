import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

data = np.load(BASE_DIR/"model_weights.npz")

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

def predict(X):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)

    Z2 = A1 @ W2 + b2

    shifted_Z2 = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_Z2 = np.exp(shifted_Z2)

    probs = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)

    prediction = np.argmax(probs, axis=1)

    return prediction, probs