"""Non-IID client partitions (Sec. 3.5).

Two datasets are supported:

``synthetic``  A 10-class Gaussian-blob problem generated in memory. It needs no
               download, runs a full 60-round experiment on a laptop CPU in under
               a minute, and is meant for debugging the orchestration logic.
``mnist``      Real MNIST, downloaded once as the original idx files. Use this for
               the numbers that go in the thesis.

Both are partitioned with the same symmetric-Dirichlet scheme, so a run can be
moved from one to the other by changing a single config field.
"""
from __future__ import annotations

import gzip
import os
import urllib.request

import numpy as np

MNIST_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _download(name: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    if os.path.exists(path):
        return path
    last = None
    for base in MNIST_URLS:
        try:
            urllib.request.urlretrieve(base + name, path)
            return path
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
    raise RuntimeError(f"could not download {name}: {last}")


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    shape = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8).reshape(shape)


def load_mnist(root: str = "data"):
    paths = {k: _download(v, root) for k, v in MNIST_FILES.items()}
    xtr = _read_idx(paths["train_x"]).astype(np.float32).reshape(-1, 784) / 255.0
    ytr = _read_idx(paths["train_y"]).astype(np.int64)
    xte = _read_idx(paths["test_x"]).astype(np.float32).reshape(-1, 784) / 255.0
    yte = _read_idx(paths["test_y"]).astype(np.int64)
    return (xtr - 0.1307) / 0.3081, ytr, (xte - 0.1307) / 0.3081, yte


def load_synthetic(num_classes: int, seed: int, dim: int = 12, per_class: int = 800):
    rng = np.random.default_rng(seed)
    # Overlapping blobs on purpose: well-separated classes saturate within a few
    # rounds and the accuracy term of the reward goes dead, which hides policy
    # differences instead of exposing them.
    centres = rng.normal(0, 0.55, size=(num_classes, dim)).astype(np.float32)
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centres[c] + rng.normal(0, 1.0, size=(per_class, dim)).astype(np.float32))
        ys.append(np.full(per_class, c, dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    cut = int(0.8 * len(x))
    return x[:cut], y[:cut], x[cut:], y[cut:]


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, seed: int,
                        min_samples: int, max_samples: int) -> list[np.ndarray]:
    """Symmetric-Dirichlet label partition.

    Draws one class-proportion vector per client and fills each client's quota by
    sampling label-wise without replacement, which keeps per-client sample counts
    heterogeneous (n_i varies) as well as the label mix.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    pool = {c: list(rng.permutation(np.where(y == c)[0])) for c in classes}
    sizes = rng.integers(min_samples, max_samples + 1, size=num_clients)
    props = rng.dirichlet(np.full(len(classes), alpha), size=num_clients)

    parts: list[np.ndarray] = []
    for i in range(num_clients):
        want = (props[i] * sizes[i]).astype(int)
        want[want.argmax()] += sizes[i] - want.sum()
        taken: list[int] = []
        for ci, c in enumerate(classes):
            k = min(want[ci], len(pool[c]))
            for _ in range(k):
                taken.append(pool[c].pop())
        if not taken:  # a client that lost its whole quota still needs one sample
            c = int(rng.choice(classes))
            if pool[c]:
                taken.append(pool[c].pop())
        arr = np.array(taken, dtype=np.int64)
        # `taken` is built by iterating classes in order, so it is class-grouped
        # (all of class 0's samples, then class 1's, ...). Shuffle it here so
        # any later code that takes a prefix of a client's partition (e.g. the
        # local-loss probe in engine.py) sees a sample representative of the
        # client's actual class mix rather than just its first class.
        rng.shuffle(arr)
        parts.append(arr)
    return parts


def class_sketch(y: np.ndarray, parts: list[np.ndarray], num_classes: int) -> np.ndarray:
    """c_{i,k}: the privacy-preserving class-distribution sketch of Sec. 3.5.

    Only normalised class proportions leave the client, never raw data.
    """
    sk = np.zeros((len(parts), num_classes), dtype=np.float64)
    for i, idx in enumerate(parts):
        if len(idx) == 0:
            continue
        counts = np.bincount(y[idx], minlength=num_classes).astype(np.float64)
        sk[i] = counts / counts.sum()
    return sk


def build(cfg) -> dict:
    d = cfg.data
    if d.dataset == "mnist":
        xtr, ytr, xte, yte = load_mnist()
        num_classes = 10
    elif d.dataset == "synthetic":
        xtr, ytr, xte, yte = load_synthetic(d.num_classes, d.seed)
        num_classes = d.num_classes
    else:
        raise ValueError(f"unknown dataset {d.dataset}")

    parts = dirichlet_partition(ytr, d.num_clients, d.dirichlet_alpha, d.seed,
                                d.min_samples, d.max_samples)
    return {
        "x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte,
        "parts": parts,
        "sketch": class_sketch(ytr, parts, num_classes),
        "n_i": np.array([len(p) for p in parts], dtype=np.float64),
        "num_classes": num_classes,
        "dim": xtr.shape[1],
    }
