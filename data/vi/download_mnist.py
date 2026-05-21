"""Download MNIST + Fashion-MNIST via torchvision and dump flat tensors.

Run this once from the project root with the project venv activated:

    python data/vi/download_mnist.py

It writes `data/vi/mnist.pt` containing a dict of float32 / int64 tensors
ready for `torch.load(...)`. The `vi.qmd` chapter loads from that file and
performs no network IO during `quarto render`.
"""

import os

# The python.org / venv Python on macOS does not trust system CAs out of the
# box. Point ssl at certifi's bundle BEFORE importing torchvision so its
# urlopen() picks it up.
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import torch
from torchvision.datasets import MNIST, FashionMNIST

OUT_DIR = "data/vi"
TV_DIR = "data/vi/torchvision"
os.makedirs(TV_DIR, exist_ok=True)


def flat(ds):
    x = ds.data.float().reshape(ds.data.shape[0], -1) / 255.0
    return x, ds.targets.long()


def main():
    mnist_train = MNIST(root=TV_DIR, train=True, download=True)
    mnist_test = MNIST(root=TV_DIR, train=False, download=True)
    fashion = FashionMNIST(root=TV_DIR, train=False, download=True)

    x_train, y_train = flat(mnist_train)
    x_test, y_test = flat(mnist_test)
    x_ood, _ = flat(fashion)

    out_path = f"{OUT_DIR}/mnist.pt"
    torch.save(
        {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "x_ood": x_ood,
        },
        out_path,
    )
    print(f"Saved {out_path} (x_train: {tuple(x_train.shape)})")


if __name__ == "__main__":
    main()
