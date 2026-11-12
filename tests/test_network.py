import numpy as np

from digit_classifier.data import one_hot
from digit_classifier.network import NeuralNetwork, softmax
from digit_classifier.train import train


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.randn(5, 10) * 50)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_gradients_match_numerical_estimate():
    rng = np.random.default_rng(0)
    net = NeuralNetwork(6, 5, 3, seed=1)
    X = rng.normal(size=(4, 6))
    Y = one_hot(rng.integers(0, 3, 4), 3)
    grads = net.gradients(X, Y)

    eps = 1e-5
    for name, param in net.params().items():
        flat = param.reshape(-1)
        for j in range(min(5, flat.size)):
            old = flat[j]
            flat[j] = old + eps
            up = net.loss(net.predict_proba(X), Y)
            flat[j] = old - eps
            down = net.loss(net.predict_proba(X), Y)
            flat[j] = old
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - grads[name].reshape(-1)[j]) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    net = NeuralNetwork(64, 16, 10, seed=3)
    net.save(tmp_path / "m.npz")
    loaded = NeuralNetwork.load(tmp_path / "m.npz")
    X = np.random.rand(3, 64)
    assert np.allclose(net.predict_proba(X), loaded.predict_proba(X))


def test_training_reaches_good_accuracy():
    _, history = train(epochs=20, verbose=False)
    assert history["test_acc"][-1] > 0.9
