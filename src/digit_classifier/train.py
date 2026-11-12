"""Train the digit classifier:  python -m digit_classifier.train"""
import argparse
from pathlib import Path

import numpy as np

from digit_classifier.data import load_data, one_hot
from digit_classifier.network import NeuralNetwork


def accuracy(net: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> float:
    return float((net.predict(X) == y).mean())


def train(epochs=60, lr=0.1, hidden=64, batch_size=32, seed=42, verbose=True):
    X_train, X_test, y_train, y_test = load_data(seed=seed)
    Y_train, Y_test = one_hot(y_train), one_hot(y_test)
    net = NeuralNetwork(X_train.shape[1], hidden, 10, seed=seed)
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            net.step(net.gradients(X_train[idx], Y_train[idx]), lr)

        history["train_loss"].append(net.loss(net.predict_proba(X_train), Y_train))
        history["test_loss"].append(net.loss(net.predict_proba(X_test), Y_test))
        history["train_acc"].append(accuracy(net, X_train, y_train))
        history["test_acc"].append(accuracy(net, X_test, y_test))
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"epoch {epoch:3d} | loss {history['train_loss'][-1]:.4f} "
                f"| train acc {history['train_acc'][-1]:.3f} | test acc {history['test_acc'][-1]:.3f}"
            )
    return net, history


def plot_history(history: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set(title="Loss", xlabel="epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set(title="Accuracy", xlabel="epoch")
    ax2.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="models/digits_mlp.npz")
    p.add_argument("--plot-path", default="docs/training_curve.png")
    a = p.parse_args()

    net, history = train(a.epochs, a.lr, a.hidden, a.batch_size, a.seed)
    net.save(a.model_path)
    plot_history(history, a.plot_path)
    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.2%}")
    print(f"Saved model to {a.model_path} and plot to {a.plot_path}")


if __name__ == "__main__":
    main()
