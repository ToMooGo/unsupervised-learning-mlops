"""Datasets used in the project.

* **Digits** (``sklearn.datasets.load_digits``): 1,797 grayscale 8x8 images, pixel values 0-16.
  A 75/25 split gives the 1,347 training / 450 test images used in Geron, Ch. 9.
* **Synthetic anomaly benchmarks**: the book's Gaussian-blob and moons datasets, with a known
  fraction of injected outliers so that detectors can be scored with precision / recall.
* **Corrupted digits**: known-bad inputs (noise, inverted, shuffled, rotated, mirrored, shifted,
  mild Gaussian noise, dimmed) used to measure how
  well the digit anomaly detector protects the deployed classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split

N_PIXELS = 64
PIXEL_MAX = 16.0
CORRUPTIONS = (
    "noise",
    "inverted",
    "shuffled",
    "rotated",
    "mirrored",
    "shifted",
    "gaussian",
    "dimmed",
)


@dataclass
class DigitsSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def sizes(self) -> dict:
        return {"n_train": int(len(self.X_train)), "n_test": int(len(self.X_test))}


def load_digits_split(test_size: float = 0.25, random_state: int = 42) -> DigitsSplit:
    """Load the digits dataset and split it as in the book (``train_test_split`` defaults)."""
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return DigitsSplit(X_train, X_test, y_train, y_test)


# --------------------------------------------------------------------------------------
# Synthetic data with injected outliers
# --------------------------------------------------------------------------------------
_SHEAR = np.array([[0.374, 0.95], [0.732, 0.598]])  # linear map used for the book's blobs
BLOB_COMPONENTS = [
    # (weight, mean, covariance) - three clusters with weights 0.4 / 0.4 / 0.2 (Geron p. 263)
    (0.4, np.array([4.0, -4.0]) @ _SHEAR, _SHEAR.T @ _SHEAR),
    (0.4, np.array([0.0, 0.0]) @ _SHEAR, _SHEAR.T @ _SHEAR),
    (0.2, np.array([3.0, -2.5]), np.eye(2) * 0.5),
]


def _n_outliers(n_inliers: int, contamination: float) -> int:
    return int(round(n_inliers * contamination / (1.0 - contamination)))


def _uniform_box(X: np.ndarray, n: int, rng: np.random.Generator, pad: float = 0.25) -> np.ndarray:
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    return rng.uniform(lo - pad * span, hi + pad * span, size=(n, X.shape[1]))


def make_blobs_with_outliers(
    n_samples: int = 1000,
    contamination: float = 0.04,
    min_mahalanobis: float = 3.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Three Gaussian clusters plus uniformly scattered outliers.

    Outliers are rejection-sampled so that each one lies at least ``min_mahalanobis`` standard
    deviations from every cluster - i.e. they are genuinely anomalous, not just unlucky inliers.

    Returns ``X`` and ``is_outlier`` (1 = injected outlier, 0 = inlier).
    """
    rng = np.random.default_rng(random_state)
    weights = np.array([w for w, _, _ in BLOB_COMPONENTS])
    counts = rng.multinomial(n_samples, weights / weights.sum())
    inliers = np.vstack(
        [
            rng.multivariate_normal(mean, cov, size=c)
            for (_, mean, cov), c in zip(BLOB_COMPONENTS, counts, strict=True)
        ]
    )
    inv_covs = [np.linalg.inv(cov) for _, _, cov in BLOB_COMPONENTS]

    def far_enough(P: np.ndarray) -> np.ndarray:
        d2 = np.stack(
            [
                np.einsum("ij,jk,ik->i", P - mean, inv, P - mean)
                for (_, mean, _), inv in zip(BLOB_COMPONENTS, inv_covs, strict=True)
            ]
        )
        return np.sqrt(d2.min(axis=0)) >= min_mahalanobis

    outliers = _rejection_sample(inliers, _n_outliers(n_samples, contamination), far_enough, rng)
    return _stack_and_shuffle(inliers, outliers, rng)


def make_moons_with_outliers(
    n_samples: int = 1000,
    noise: float = 0.05,
    contamination: float = 0.04,
    min_distance: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """The moons dataset (Geron p. 257) plus outliers at least ``min_distance`` from any inlier."""
    rng = np.random.default_rng(random_state)
    inliers, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    def far_enough(P: np.ndarray) -> np.ndarray:
        d = np.sqrt(((P[:, None, :] - inliers[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return d >= min_distance

    outliers = _rejection_sample(inliers, _n_outliers(n_samples, contamination), far_enough, rng)
    return _stack_and_shuffle(inliers, outliers, rng)


def _rejection_sample(inliers, n_needed, accept, rng, max_rounds: int = 200) -> np.ndarray:
    kept: list[np.ndarray] = []
    total = 0
    for _ in range(max_rounds):
        cand = _uniform_box(inliers, max(4 * n_needed, 64), rng)
        cand = cand[accept(cand)]
        kept.append(cand)
        total += len(cand)
        if total >= n_needed:
            return np.vstack(kept)[:n_needed]
    raise RuntimeError("Could not sample enough outliers; relax the distance constraint.")


def _stack_and_shuffle(inliers, outliers, rng) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers), dtype=int), np.ones(len(outliers), dtype=int)]
    order = rng.permutation(len(X))
    return X[order], y[order]


# --------------------------------------------------------------------------------------
# Corrupted digits (known anomalies for the deployed detector)
# --------------------------------------------------------------------------------------
CORRUPTION_DESCRIPTIONS = {
    "noise": "pure random pixels (0-16)",
    "inverted": "16 - pixel (white digit on black)",
    "shuffled": "pixels randomly permuted",
    "rotated": "rotated 90 degrees",
    "mirrored": "flipped left-right",
    "shifted": "moved 1 pixel to the right",
    "gaussian": "Gaussian pixel noise, sigma = 3",
    "dimmed": "all pixels x 0.5 (faint stroke)",
}


def corrupt_digits(X: np.ndarray, kind: str, random_state: int = 0) -> np.ndarray:
    """Return a corrupted copy of 8x8 digit images (rows of 64 pixels in 0..16).

    The list deliberately mixes obvious corruptions (noise, inverted, shuffled) with subtle
    ones (mirrored, shifted, mild noise, dimmed) so the evaluation also shows the detector's
    blind spots, not only easy wins.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    imgs = X.reshape(-1, 8, 8)
    if kind == "noise":
        return rng.integers(0, int(PIXEL_MAX) + 1, size=X.shape).astype(float)
    if kind == "inverted":
        return PIXEL_MAX - X
    if kind == "shuffled":
        return np.stack([row[rng.permutation(N_PIXELS)] for row in X])
    if kind == "rotated":
        return np.rot90(imgs, axes=(1, 2)).reshape(-1, N_PIXELS)
    if kind == "mirrored":
        return imgs[:, :, ::-1].reshape(-1, N_PIXELS)
    if kind == "shifted":
        out = np.zeros_like(imgs)
        out[:, :, 1:] = imgs[:, :, :-1]
        return out.reshape(-1, N_PIXELS)
    if kind == "gaussian":
        return np.clip(X + rng.normal(0.0, 3.0, X.shape), 0.0, PIXEL_MAX)
    if kind == "dimmed":
        return X * 0.5
    raise ValueError(f"Unknown corruption {kind!r}; choose from {CORRUPTIONS}")
