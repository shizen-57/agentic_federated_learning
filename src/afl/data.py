"""Non-IID client partitions (Sec. 3.5).

Four datasets are supported:

``synthetic``  A 10-class Gaussian-blob problem generated in memory. It needs no
               download, runs a full 60-round experiment on a laptop CPU in under
               a minute, and is meant for debugging the orchestration logic.
``mnist``      Real MNIST, downloaded once as the original idx files.
``cifar10``    CIFAR-10 colour images (10 classes, 3x32x32), downloaded once from
               the official Toronto host and cached as uint8 arrays.
``femnist``    LEAF's FEMNIST: handwritten digits and letters (62 classes) from
               ~3,600 real writers. Run ``scripts/prepare_femnist.py`` once first.

MNIST, CIFAR-10 and synthetic are partitioned with the same symmetric-Dirichlet
scheme, so a run can be moved between them by changing a single config field.
FEMNIST uses its *natural* partition instead -- one client per real writer -- so
its non-IID-ness comes from real people, not from an artificial Dirichlet draw.
"""
from __future__ import annotations

import glob
import gzip
import io
import os
import pickle
import tarfile
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
CIFAR_URLS = [
    "https://cave.cs.toronto.edu/kriz/",
    "https://www.cs.toronto.edu/~kriz/",
]
CIFAR_FILE = "cifar-10-python.tar.gz"
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
FEMNIST_FILES = ("femnist_x.npy", "femnist_y.npy", "femnist_writer.npy")


def _progress_hook(name: str):
    state = {"next": 0}

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(100 * blocks * block_size / total))
        if pct >= state["next"]:
            print(f"  downloading {name}: {pct}% of {total / 1e6:.0f} MB", flush=True)
            state["next"] = pct + 10
    return hook


def _download(name: str, root: str, bases: list[str] = MNIST_URLS) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    if os.path.exists(path):
        return path
    last = None
    tmp = path + ".part"
    for base in bases:
        try:
            urllib.request.urlretrieve(base + name, tmp, _progress_hook(name))
            os.replace(tmp, path)
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


class _ArrayOnlyUnpickler(pickle.Unpickler):
    """CIFAR-10 ships as pickles; refuse anything but the numpy array globals.

    A plain ``pickle.load`` on a downloaded file can run arbitrary code. The
    official batches only need numpy's array reconstruction, so that is all
    this unpickler will resolve.
    """

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from CIFAR-10 batch")
        return super().find_class(module, name)


def _unpickle_batch(raw: bytes):
    d = _ArrayOnlyUnpickler(io.BytesIO(raw), encoding="bytes").load()
    return (np.asarray(d[b"data"], dtype=np.uint8), np.asarray(d[b"labels"], dtype=np.int64))


def _assemble_cifar(batches: dict) -> dict:
    return {
        "xtr": np.concatenate([batches[f"data_batch_{i}"][0] for i in range(1, 6)]),
        "ytr": np.concatenate([batches[f"data_batch_{i}"][1] for i in range(1, 6)]),
        "xte": batches["test_batch"][0], "yte": batches["test_batch"][1],
    }


def _cifar_from_dir(d: str) -> dict:
    """Read the six official python batch files from a directory."""
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    out = {}
    for n in names:
        with open(os.path.join(d, n), "rb") as fh:
            out[n] = _unpickle_batch(fh.read())
    return _assemble_cifar(out)


def _find_cifar_dir(root: str) -> str | None:
    """A directory holding the official python batches, if one is already here.

    Looks in ``root`` (an unpacked ``cifar-10-batches-py``) and, on Kaggle, in
    any attached dataset under /kaggle/input -- much faster than downloading
    from Toronto, whose server can take 20+ minutes.
    """
    here = os.path.join(root, "cifar-10-batches-py")
    if os.path.exists(os.path.join(here, "data_batch_1")):
        return here
    if os.path.isdir("/kaggle/input"):
        for hit in sorted(glob.glob("/kaggle/input/**/data_batch_1", recursive=True)):
            d = os.path.dirname(hit)
            if all(os.path.exists(os.path.join(d, n))
                   for n in [f"data_batch_{i}" for i in range(2, 6)] + ["test_batch"]):
                return d
    return None


def _cifar_cache(root: str) -> str:
    """Build the uncompressed uint8 .npz once, from a local copy or the official tarball."""
    cache = os.path.join(root, "cifar10_uint8.npz")
    if os.path.exists(cache):
        return cache
    os.makedirs(root, exist_ok=True)
    local = _find_cifar_dir(root)
    if local:
        print(f"CIFAR-10: using the copy found at {local}", flush=True)
        arrays = _cifar_from_dir(local)
    else:
        print("CIFAR-10: no local copy found, downloading from the official site "
              "(can be slow; on Kaggle, add a 'cifar10-python' dataset as input instead)",
              flush=True)
        tar_path = _download(CIFAR_FILE, root, CIFAR_URLS)
        batches = {}
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf.getmembers():
                base = os.path.basename(m.name)
                if base.startswith("data_batch_") or base == "test_batch":
                    batches[base] = _unpickle_batch(tf.extractfile(m).read())
        arrays = _assemble_cifar(batches)
    tmp = cache + ".part.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, cache)
    return cache


def load_cifar10(root: str = "data"):
    """CIFAR-10 as flat 3072-vectors in CHW order, normalised per channel."""
    z = np.load(_cifar_cache(root))

    def prep(x):
        x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        x = (x - CIFAR_MEAN[None, :, None, None]) / CIFAR_STD[None, :, None, None]
        return x.reshape(len(x), -1)

    return prep(z["xtr"]), z["ytr"], prep(z["xte"]), z["yte"]


def load_femnist(root: str, num_clients: int, seed: int, min_samples: int,
                 max_samples: int, test_frac: float = 0.1, return_owner: bool = False):
    """Natural per-writer partition of FEMNIST (LEAF).

    Picks ``num_clients`` writers with at least ``min_samples`` images, caps each
    at ``max_samples``, and holds out ``test_frac`` of every chosen writer's data
    for a pooled test set -- LEAF's standard per-writer train/test split. The
    writers chosen depend on the seed, so different seeds see different people.
    """
    paths = [os.path.join(root, f) for f in FEMNIST_FILES]
    if not all(os.path.exists(p) for p in paths):
        raise FileNotFoundError(
            "FEMNIST is not prepared. Run once: python scripts/prepare_femnist.py")
    x_all = np.load(paths[0], mmap_mode="r")
    y_all = np.load(paths[1])
    w_all = np.load(paths[2])

    order = np.argsort(w_all, kind="stable")
    counts = np.bincount(w_all)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    eligible = np.where(counts >= min_samples)[0]
    if len(eligible) < num_clients:
        raise ValueError(f"only {len(eligible)} FEMNIST writers have >= {min_samples} "
                         f"samples; asked for {num_clients}")
    rng = np.random.default_rng(seed)
    writers = rng.choice(eligible, size=num_clients, replace=False)

    train_idx, test_idx, parts, owner, off = [], [], [], [], 0
    for k, w in enumerate(writers):
        idx = order[starts[w]:starts[w] + counts[w]].copy()
        rng.shuffle(idx)
        idx = idx[:max_samples]
        n_test = max(1, int(round(test_frac * len(idx))))
        test_idx.append(idx[:n_test])
        owner.append(np.full(n_test, k, dtype=np.int64))
        tr = idx[n_test:]
        train_idx.append(tr)
        parts.append(np.arange(off, off + len(tr), dtype=np.int64))
        off += len(tr)
    tr_all = np.concatenate(train_idx)
    te_all = np.concatenate(test_idx)
    # Fancy indexing on a memmap reads only the rows needed; sorting first keeps
    # the read sequential, and the inverse permutation restores client order.
    def gather(ix):
        srt = np.argsort(ix)
        out = np.empty((len(ix), x_all.shape[1]), dtype=np.float32)
        out[srt] = x_all[ix[srt]].astype(np.float32) / 255.0
        return out
    xtr, xte = gather(tr_all), gather(te_all)
    mu, sd = float(xtr.mean()), float(xtr.std()) + 1e-6
    out = ((xtr - mu) / sd, y_all[tr_all].astype(np.int64),
           (xte - mu) / sd, y_all[te_all].astype(np.int64), parts)
    # owner[j] = the client whose held-out writing test example j is.
    return out + (np.concatenate(owner),) if return_owner else out


def ensure(dataset: str, root: str = "data") -> None:
    """Fetch/prepare a dataset once, before parallel workers race to do it."""
    if dataset == "mnist":
        for name in MNIST_FILES.values():
            _download(name, root)
    elif dataset == "cifar10":
        _cifar_cache(root)
    elif dataset == "femnist":
        if not all(os.path.exists(os.path.join(root, f)) for f in FEMNIST_FILES):
            raise FileNotFoundError(
                "FEMNIST is not prepared. Run once: python scripts/prepare_femnist.py")


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
    """c_{i,k}: the class-distribution sketch of Sec. 3.5.

    Only normalised class proportions leave the client, never raw data -- but
    the proportions themselves reveal the client's label mix to the server,
    so the sketch is not privacy-preserving on its own.
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
    parts = None
    owner = None
    if d.dataset == "mnist":
        xtr, ytr, xte, yte = load_mnist()
        num_classes = 10
    elif d.dataset == "cifar10":
        xtr, ytr, xte, yte = load_cifar10()
        num_classes = 10
    elif d.dataset == "femnist":
        xtr, ytr, xte, yte, parts, owner = load_femnist(
            "data", d.num_clients, d.seed, d.min_samples, d.max_samples, return_owner=True)
        num_classes = 62
    elif d.dataset == "synthetic":
        xtr, ytr, xte, yte = load_synthetic(d.num_classes, d.seed)
        num_classes = d.num_classes
    else:
        raise ValueError(f"unknown dataset {d.dataset}")

    if parts is None:
        parts = dirichlet_partition(ytr, d.num_clients, d.dirichlet_alpha, d.seed,
                                    d.min_samples, d.max_samples)
    return {
        "x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte,
        "parts": parts,
        # Which client each test example belongs to (FEMNIST: the writer's own
        # held-out images); -1 everywhere for a pooled test set.
        "test_owner": owner if owner is not None else np.full(len(yte), -1, dtype=np.int64),
        "sketch": class_sketch(ytr, parts, num_classes),
        "n_i": np.array([len(p) for p in parts], dtype=np.float64),
        "num_classes": num_classes,
        "dim": xtr.shape[1],
    }
