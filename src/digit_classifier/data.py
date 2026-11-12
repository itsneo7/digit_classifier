"""Load and prepare the handwritten digits dataset (bundled with scikit-learn, no download)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def load_data(test_size: float = 0.2, seed: int = 42):
    """Return X_train, X_test, y_train, y_test. Pixels are scaled to [0, 1]."""
    digits = load_digits()
    X = digits.data / 16.0  # raw pixel values are 0..16
    y = digits.target
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
