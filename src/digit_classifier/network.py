"""A small fully-connected neural network implemented from scratch with NumPy."""
from pathlib import Path

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """Input -> Dense(hidden) + ReLU -> Dense(classes) + Softmax."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, num_classes: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation works well with ReLU
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes)
        self._velocity = {k: np.zeros_like(v) for k, v in self.params().items()}

    def params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        probs = softmax(a1 @ self.W2 + self.b2)
        return z1, a1, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[2]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @staticmethod
    def loss(probs: np.ndarray, Y: np.ndarray) -> float:
        """Mean cross-entropy loss; Y is one-hot."""
        return float(-np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1)))

    def gradients(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Backpropagation for softmax + cross-entropy."""
        n = X.shape[0]
        z1, a1, probs = self.forward(X)
        dz2 = (probs - Y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads: dict, lr: float, momentum: float = 0.9) -> None:
        """SGD with momentum (updates parameters in place)."""
        for name, param in self.params().items():
            self._velocity[name] = momentum * self._velocity[name] - lr * grads[name]
            param += self._velocity[name]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.params())

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        data = np.load(path)
        net = cls(data["W1"].shape[0], data["W1"].shape[1], data["W2"].shape[1])
        for name, param in net.params().items():
            param[...] = data[name]
        return net
