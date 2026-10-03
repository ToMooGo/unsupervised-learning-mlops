"""Unit tests for the three ML parts (small, fast settings)."""

import numpy as np
import pytest
from sklearn.base import clone

from unsupervised_mlops.anomaly import (
    DBSCANAnomalyDetector,
    GaussianMixtureAnomalyDetector,
    benchmark_detectors,
    calibrate_dbscan_eps,
    evaluate_on_corruptions,
    select_gmm,
)
from unsupervised_mlops.clustering_features import build_kmeans_logreg_pipeline, tune_n_clusters
from unsupervised_mlops.data import make_blobs_with_outliers, make_moons_with_outliers
from unsupervised_mlops.semi_supervised import (
    active_learning,
    find_representatives,
    partial_propagation_mask,
    propagate_labels,
    resolve_representative_labels,
    run_part_b,
)


# ---------------------------------------------------------------- Part A
@pytest.mark.parametrize("scale", [False, True])
def test_kmeans_pipeline_outputs_distances_to_k_centroids(split, scale):
    pipe = build_kmeans_logreg_pipeline(n_clusters=12, n_init=2, scale_distances=scale).fit(
        split.X_train, split.y_train
    )
    assert pipe.named_steps["kmeans"].transform(split.X_test[:3]).shape == (3, 12)
    assert pipe.score(split.X_test, split.y_test) > 0.85
    assert ("scaler" in pipe.named_steps) is scale


def test_grid_search_returns_a_k_from_the_grid(split):
    search = tune_n_clusters(
        split.X_train, split.y_train, [10, 30], cv=2, n_jobs=1, n_init=1, scale_distances=True
    )
    assert search.best_params_["kmeans__n_clusters"] in (10, 30)
    assert "scaler" in search.best_estimator_.named_steps


# ---------------------------------------------------------------- Part B
def test_representatives_are_closest_to_each_centroid(split):
    km, dist, rep = find_representatives(split.X_train, k=20, n_init=2)
    assert len(rep) == 20 and len(set(rep.tolist())) == 20
    for c in range(20):
        assert dist[rep[c], c] == pytest.approx(dist[:, c].min())


def test_propagation_and_partial_mask(split):
    km, dist, rep = find_representatives(split.X_train, k=20, n_init=2)
    y_prop = propagate_labels(km.labels_, split.y_train[rep])
    assert y_prop.shape == split.y_train.shape
    mask = partial_propagation_mask(dist, km.labels_, 20)
    assert 0.15 < mask.mean() < 0.3  # ~20 % of each cluster
    # labels close to centroids are more reliable than the average propagated label
    assert np.mean(y_prop[mask] == split.y_train[mask]) >= np.mean(y_prop == split.y_train)


def test_human_labels_override_oracle_and_fail_when_missing():
    rep = np.array([3, 7])
    oracle = np.arange(10)
    labels, source = resolve_representative_labels(rep, oracle, {3: 9, 7: 1})
    assert labels.tolist() == [9, 1] and source == "human"
    with pytest.raises(ValueError):
        resolve_representative_labels(rep, oracle, {3: 9})
    labels, source = resolve_representative_labels(rep, oracle, {3: 9}, allow_oracle_fallback=True)
    assert labels.tolist() == [9, 7] and source == "human+oracle"


def test_part_b_beats_random_labels(split):
    cfg = {"n_labeled": 50, "percentile_closest": 20, "percentile_sweep": [20, 100], "n_init": 3}
    res = run_part_b(split, cfg)
    m = res.metrics
    assert m["partial_propagation_accuracy"] > m["random50_accuracy"] + 0.05
    assert m["partial_propagation_label_accuracy"] > 0.95
    assert len(res.representatives["train_index"]) == 50


def test_active_learning_adds_queries(split):
    mask = np.zeros(len(split.X_train), bool)
    mask[:30] = True
    y = split.y_train.copy()
    df = active_learning(
        split.X_train,
        split.y_train,
        mask,
        y,
        split.X_test,
        split.y_test,
        n_human_start=30,
        rounds=2,
        queries_per_round=5,
    )
    assert df.human_labels.tolist() == [30, 35, 40]


# ---------------------------------------------------------------- Part C
def test_gmm_detector_flags_the_requested_percentile():
    X, _ = make_blobs_with_outliers(n_samples=600, random_state=0)
    det = GaussianMixtureAnomalyDetector(
        n_components=3, threshold_percentile=4, n_init=2, random_state=0
    ).fit(X)
    flagged = det.predict(X) == -1
    assert abs(flagged.mean() - 0.04) < 0.01
    assert np.allclose(det.decision_function(X), det.score_samples(X) - det.threshold_)
    clone(det)  # scikit-learn compatible


def test_holdout_calibration_reduces_false_positives_on_unseen_digits(split):
    common = dict(
        n_components=6, threshold_percentile=4, pca_variance=0.95, n_init=2, random_state=0
    )
    naive = GaussianMixtureAnomalyDetector(**common).fit(split.X_train)
    calibrated = GaussianMixtureAnomalyDetector(**common, calibration_fraction=0.25).fit(
        split.X_train
    )
    fp_naive = np.mean(naive.predict(split.X_test) == -1)
    fp_cal = np.mean(calibrated.predict(split.X_test) == -1)
    assert fp_cal < fp_naive


def test_digit_detector_catches_obvious_corruptions(split):
    det = GaussianMixtureAnomalyDetector(6, "full", 4, 0.95, 0.25, n_init=2, random_state=0).fit(
        split.X_train
    )
    table, summary = evaluate_on_corruptions(det, split.X_test)
    rates = dict(zip(table.input, table.flagged_pct, strict=True))
    for kind in ("noise", "inverted", "shuffled"):
        assert rates[kind] > 95
    assert rates["clean test digits"] < 10
    assert summary["roc_auc_clean_vs_corrupted"] > 0.9


def test_select_gmm_prefers_three_components_on_three_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(150, 2)) for c in ([0, 0], [4, 4], [0, 5])])
    best, table = select_gmm(X, [1, 2, 3, 4, 5], ["full"], n_init=2)
    assert best["n_components"] == 3
    assert {"bic", "aic"} <= set(table.columns)


def test_dbscan_detector_and_eps_calibration():
    X, y = make_moons_with_outliers(n_samples=400, random_state=0)
    eps, table = calibrate_dbscan_eps(X, target_fraction=0.04)
    assert eps > 0 and len(table) > 10
    det = DBSCANAnomalyDetector(eps=eps).fit(X)
    flagged = det.labels_ == -1
    assert flagged[y == 1].mean() > 0.8
    far = np.array([[10.0, 10.0]])
    assert det.predict(far)[0] == -1


@pytest.mark.parametrize("maker", [make_blobs_with_outliers, make_moons_with_outliers])
def test_benchmark_table(maker):
    X, y = maker(n_samples=300, random_state=0)
    table, art = benchmark_detectors(X, y, 0.04, gmm_k_values=[1, 2, 3, 4], gmm_n_init=1)
    assert {"GMM (BIC-selected)", "DBSCAN"} <= set(table.detector)
    assert table.loc[table.detector == "GMM (BIC-selected)", "f1"].iloc[0] > 0.6
    assert "gmm" in art and "dbscan" in art
