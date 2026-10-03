import numpy as np
import pytest

from unsupervised_mlops.config import deep_update, k_values, load_config
from unsupervised_mlops.data import (
    BLOB_COMPONENTS,
    CORRUPTIONS,
    corrupt_digits,
    make_blobs_with_outliers,
    make_moons_with_outliers,
)


def test_digits_split_matches_book(split):
    assert split.sizes == {"n_train": 1347, "n_test": 450}
    assert split.X_train.shape[1] == 64
    assert split.X_train.min() >= 0 and split.X_train.max() <= 16


@pytest.mark.parametrize("maker", [make_blobs_with_outliers, make_moons_with_outliers])
def test_injected_outlier_fraction(maker):
    X, y = maker(n_samples=500, contamination=0.04, random_state=1)
    assert len(X) == len(y) == 500 + round(500 * 0.04 / 0.96)
    assert set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - 0.04) < 0.005


def test_blob_outliers_are_far_from_every_cluster():
    X, y = make_blobs_with_outliers(n_samples=400, min_mahalanobis=3.0, random_state=3)
    for _, mean, cov in BLOB_COMPONENTS:
        inv = np.linalg.inv(cov)
        d = np.sqrt(np.einsum("ij,jk,ik->i", X[y == 1] - mean, inv, X[y == 1] - mean))
        assert (d >= 3.0).all()


def test_moon_outliers_keep_their_distance():
    X, y = make_moons_with_outliers(n_samples=300, min_distance=0.15, random_state=0)
    inl, out = X[y == 0], X[y == 1]
    d = np.sqrt(((out[:, None] - inl[None]) ** 2).sum(-1)).min(1)
    assert (d >= 0.15).all()


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_corruptions_keep_shape_and_range(split, kind):
    Xc = corrupt_digits(split.X_test[:20], kind, random_state=0)
    assert Xc.shape == (20, 64)
    assert Xc.min() >= 0 and Xc.max() <= 16
    if kind not in ("gaussian",):
        assert not np.allclose(Xc, split.X_test[:20])


def test_unknown_corruption_raises(split):
    with pytest.raises(ValueError):
        corrupt_digits(split.X_test[:2], "melted")


def test_config_env_substitution_and_extends(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text(
        "a: 1\nnested: {x: 1, y: 2}\nuri: ${MY_URI:sqlite:///x.db}\nflag: ${MY_FLAG:false}\n"
    )
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nnested: {y: 3}\n")
    monkeypatch.setenv("MY_FLAG", "true")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"a": 1, "nested": {"x": 1, "y": 3}, "uri": "sqlite:///x.db", "flag": True}


def test_repo_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    full = load_config(root / "full_flow_config.yaml")
    quick = load_config(root / "quick_flow_config.yaml")
    assert full["flow"] == "full" and quick["part_a"]["robustness"]["enabled"] is False
    assert quick["part_b"]["n_labeled"] == 50  # inherited


def test_helpers():
    assert k_values({"start": 2, "stop": 10, "step": 4}) == [2, 6]
    assert k_values([3, 5]) == [3, 5]
    assert deep_update({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
