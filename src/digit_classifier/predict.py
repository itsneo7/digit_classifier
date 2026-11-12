"""Show predictions on random test digits:  python -m digit_classifier.predict"""
import argparse

import numpy as np

from digit_classifier.data import load_data
from digit_classifier.network import NeuralNetwork

SHADES = " .:-=+*#%@"


def ascii_digit(pixels: np.ndarray) -> str:
    img = pixels.reshape(8, 8)
    return "\n".join("".join(SHADES[int(v * 9)] * 2 for v in row) for row in img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="models/digits_mlp.npz")
    p.add_argument("-n", type=int, default=5, help="number of samples to show")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    net = NeuralNetwork.load(a.model_path)
    _, X_test, _, y_test = load_data()
    rng = np.random.default_rng(a.seed)
    for i in rng.choice(len(X_test), size=a.n, replace=False):
        probs = net.predict_proba(X_test[i : i + 1])[0]
        pred = int(probs.argmax())
        mark = "OK " if pred == y_test[i] else "WRONG"
        print(ascii_digit(X_test[i]))
        print(f"-> predicted {pred} ({probs[pred]:.0%} confident), actual {y_test[i]}  [{mark}]\n")


if __name__ == "__main__":
    main()
