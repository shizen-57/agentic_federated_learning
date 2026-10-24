#!/usr/bin/env python3
"""Convert LEAF's FEMNIST into three flat numpy arrays, once.

Source: the Flower-maintained copy of LEAF FEMNIST on Hugging Face
(flwrlabs/femnist): 814,277 28x28 grayscale images of digits and upper/lower
case letters (62 classes) written by 3,597 distinct people, with each image's
writer id kept. One writer = one client gives a *natural* non-IID partition.

Output (in data/):
    femnist_x.npy       uint8  (num_images, 784)
    femnist_y.npy       int16  (num_images,)       class 0-61
    femnist_writer.npy  int32  (num_images,)       writer index 0..3596

The arrays are saved uncompressed so afl.data can memory-map them and read only
the writers a run actually uses.

Needs pyarrow and Pillow, for this one-off conversion only (both preinstalled
on Colab and Kaggle):

    pip install pyarrow pillow
    python scripts/prepare_femnist.py
"""
from __future__ import annotations

import argparse
import io
import os
import urllib.request

import numpy as np

URL = ("https://huggingface.co/datasets/flwrlabs/femnist/resolve/main/"
       "data/train-00000-of-00001.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--parquet", default=None,
                    help="path to an already-downloaded parquet file")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    os.makedirs(args.root, exist_ok=True)
    src = args.parquet or os.path.join(args.root, "femnist.parquet")
    if not os.path.exists(src):
        print(f"downloading {URL} (~200 MB)")
        urllib.request.urlretrieve(URL, src + ".part")
        os.replace(src + ".part", src)

    pf = pq.ParquetFile(src)
    n = pf.metadata.num_rows
    x = np.empty((n, 784), dtype=np.uint8)
    y = np.empty(n, dtype=np.int16)
    writer_names: list[str] = []
    off = 0
    for batch in pf.iter_batches(batch_size=20000, columns=["image", "writer_id", "character"]):
        imgs = batch.column("image").to_pylist()
        for k, im in enumerate(imgs):
            raw = im["bytes"] if isinstance(im, dict) else im
            arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
            if arr.shape != (28, 28):
                raise ValueError(f"unexpected image shape {arr.shape} at row {off + k}")
            x[off + k] = arr.reshape(-1)
        y[off:off + len(imgs)] = np.asarray(batch.column("character").to_pylist(), dtype=np.int16)
        writer_names.extend(batch.column("writer_id").to_pylist())
        off += len(imgs)
        print(f"  {off}/{n}", flush=True)

    names, w = np.unique(np.asarray(writer_names), return_inverse=True)
    np.save(os.path.join(args.root, "femnist_x.npy"), x)
    np.save(os.path.join(args.root, "femnist_y.npy"), y)
    np.save(os.path.join(args.root, "femnist_writer.npy"), w.astype(np.int32))
    counts = np.bincount(w)
    print(f"wrote {n} images, {len(names)} writers, {int(y.max()) + 1} classes; "
          f"samples per writer: median {int(np.median(counts))}, "
          f"min {counts.min()}, max {counts.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
