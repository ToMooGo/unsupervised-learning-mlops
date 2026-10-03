"""Part C - density-based anomaly detection (Geron, Ch. 9, pp. 256-258 and 266-268).

* :class:`GaussianMixtureAnomalyDetector` - fit a Gaussian Mixture Model, flag instances whose
  log-density falls below the ``threshold_percentile``-th percentile of the training densities
  (4 % in the book: "the ratio of defective products is usually well-known").
* :func:`select_gmm` - choose the number of components and covariance type by BIC / AIC.
* :class:`DBSCANAnomalyDetector` - DBSCAN noise points are anomalies; new points are scored with
  a KNN classifier trained on the core instances plus a maximum distance (book p. 258).
* :func:`benchmark_detectors` - GMM vs DBSCAN (plus the book's other detectors as reference)
  on data with *known* injected outliers, scored with precision / recall / F1 / ROC-AUC.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.utils.validation import check_array, check_is_fitted

from .data import CORRUPTIONS, corrupt_digits

COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")


# --------------------------------------------------------------------------------------
# Gaussian Mixture detector
# --------------------------------------------------------------------------------------
class GaussianMixtureAnomalyDetector(OutlierMixin, BaseEstimator):
    """Flag the lowest-density instances under a Gaussian Mixture Model.

    Follows scikit-learn's outlier-detector conventions: ``predict`` returns ``1`` for inliers
    and ``-1`` for anomalies; ``score_samples`` is the log-density (higher = more normal);
    ``decision_function = score_samples - threshold_`` (negative = anomaly).

    ``pca_variance`` (e.g. 0.95) optionally projects the data with PCA first (Geron, Ch. 8),
    which keeps a full-covariance GMM well-posed on 64-pixel images.

    ``calibration_fraction`` > 0 holds out that share of the data to set the threshold on
    densities the GMM has *not* been fitted to. A threshold set on the training densities
    is optimistic (the model fits its own data best), so unseen normal data gets flagged
    more often than the intended percentile.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        threshold_percentile: float = 4.0,
        pca_variance: float | None = None,
        calibration_fraction: float = 0.0,
        n_init: int = 10,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.threshold_percentile = threshold_percentile
        self.pca_variance = pca_variance
        self.calibration_fraction = calibration_fraction
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _project(self, X):
        return self.pca_.transform(X) if self.pca_ is not None else X

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        X_fit, X_cal = X, X
        if self.calibration_fraction and self.calibration_fraction > 0:
            rng = np.random.default_rng(self.random_state)
            order = rng.permutation(len(X))
            n_cal = max(1, int(round(self.calibration_fraction * len(X))))
            X_cal, X_fit = X[order[:n_cal]], X[order[n_cal:]]
        self.pca_ = (
            PCA(n_components=self.pca_variance, random_state=self.random_state).fit(X_fit)
            if self.pca_variance
            else None
        )
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(self._project(X_fit))
        densities = self.gmm_.score_samples(self._project(X_cal))
        self.threshold_ = float(np.percentile(densities, self.threshold_percentile))
        self.offset_ = self.threshold_
        self.train_density_quantiles_ = {
            str(q): float(np.percentile(densities, q)) for q in (1, 4, 10, 25, 50, 75, 90, 99)
        }
        return self

    def score_samples(self, X) -> np.ndarray:
        check_is_fitted(self, "gmm_")
        return self.gmm_.score_samples(self._project(check_array(X, dtype=float)))

    def decision_function(self, X) -> np.ndarray:
        return self.score_samples(X) - self.threshold_

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def predict_cluster(self, X) -> np.ndarray:
        check_is_fitted(self, "gmm_")
        return self.gmm_.predict(self._project(check_array(X, dtype=float)))

    def density_percentile(self, X) -> np.ndarray:
        """Rough percentile of each sample's density among the training densities (0-100)."""
        qs = np.array([float(k) for k in self.train_density_quantiles_])
        vals = np.array(list(self.train_density_quantiles_.values()))
        return np.interp(self.score_samples(X), vals, qs, left=0.0, right=100.0)


def select_gmm(
    X: np.ndarray,
    k_values: list[int],
    covariance_types: tuple[str, ...] | list[str] = COVARIANCE_TYPES,
    criterion: str = "bic",
    pca_variance: float | None = None,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Fit a GMM for every (k, covariance_type) and pick the one minimising BIC (or AIC)."""
    Z = (
        PCA(n_components=pca_variance, random_state=random_state).fit_transform(X)
        if pca_variance
        else X
    )
    rows = []
    for cov in covariance_types:
        for k in k_values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm = GaussianMixture(
                    n_components=k, covariance_type=cov, n_init=n_init, random_state=random_state
                ).fit(Z)
            rows.append(
                {
                    "k": k,
                    "covariance_type": cov,
                    "bic": gm.bic(Z),
                    "aic": gm.aic(Z),
                    "converged": bool(gm.converged_),
                }
            )
    table = pd.DataFrame(rows)
    best = table.loc[table[criterion].idxmin()]
    return {"n_components": int(best.k), "covariance_type": str(best.covariance_type)}, table


# --------------------------------------------------------------------------------------
# DBSCAN detector
# --------------------------------------------------------------------------------------
class DBSCANAnomalyDetector(OutlierMixin, BaseEstimator):
    """DBSCAN noise points (label -1) are anomalies. For *new* instances, a KNN classifier trained
    on the core instances finds the nearest core point; farther than ``eps`` -> anomaly."""

    def __init__(self, eps: float = 0.2, min_samples: int = 5, n_neighbors: int = 50):
        self.eps = eps
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        self.dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.labels_ = self.dbscan_.labels_
        core = self.dbscan_.core_sample_indices_
        self.knn_ = None
        if len(core):
            self.knn_ = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(core)))
            self.knn_.fit(self.dbscan_.components_, self.labels_[core])
        return self

    def fit_predict(self, X, y=None):
        return np.where(self.fit(X).labels_ == -1, -1, 1)

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "dbscan_")
        X = check_array(X, dtype=float)
        if self.knn_ is None:
            return -np.ones(len(X), dtype=int)
        dist, _ = self.knn_.kneighbors(X, n_neighbors=1)
        return np.where(dist.ravel() > self.eps, -1, 1)


def calibrate_dbscan_eps(
    X: np.ndarray, target_fraction: float = 0.04, min_samples: int = 5, n_grid: int = 80
) -> tuple[float, pd.DataFrame]:
    """Label-free choice of ``eps``: the value whose noise fraction is closest to the expected
    contamination - the same prior knowledge (\"about 4 % are defective\") the GMM threshold uses."""
    scale = float(np.mean(X.std(axis=0)))
    rows = []
    for eps in np.geomspace(0.005, 1.5, n_grid) * scale:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
        rows.append(
            {
                "eps": float(eps),
                "noise_fraction": float(np.mean(labels == -1)),
                "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            }
        )
    table = pd.DataFrame(rows)
    best = table.iloc[(table.noise_fraction - target_fraction).abs().argsort().iloc[0]]
    return float(best.eps), table


# --------------------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------------------
def _scores(y_true, flagged, anomaly_score=None) -> dict:
    out = {
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "f1": f1_score(y_true, flagged, zero_division=0),
        "flagged_pct": 100 * float(np.mean(flagged)),
        "roc_auc": np.nan,
        "average_precision": np.nan,
    }
    if anomaly_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, anomaly_score)
        out["average_precision"] = average_precision_score(y_true, anomaly_score)
    return out


def benchmark_detectors(
    X: np.ndarray,
    y_outlier: np.ndarray,
    contamination: float = 0.04,
    gmm_k_values: list[int] | None = None,
    gmm_n_init: int = 10,
    min_samples: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Score detectors on data with known outliers. Every detector gets the same prior: the
    expected contamination. Returns the results table and fitted artefacts for plotting."""
    gmm_k_values = gmm_k_values or list(range(1, 11))
    pct = 100 * contamination
    rows, artefacts = [], {}

    best, ic_table = select_gmm(X, gmm_k_values, n_init=gmm_n_init, random_state=random_state)
    gmm = GaussianMixtureAnomalyDetector(
        **best, threshold_percentile=pct, n_init=gmm_n_init, random_state=random_state
    ).fit(X)
    gmm_flag = gmm.predict(X) == -1
    rows.append(
        {
            "detector": "GMM (BIC-selected)",
            "settings": f"k={best['n_components']}, {best['covariance_type']}",
            **_scores(y_outlier, gmm_flag, -gmm.score_samples(X)),
        }
    )
    artefacts.update(gmm=gmm, gmm_flag=gmm_flag, ic_table=ic_table)

    eps, eps_table = calibrate_dbscan_eps(X, contamination, min_samples)
    db = DBSCANAnomalyDetector(eps=eps, min_samples=min_samples).fit(X)
    db_flag = db.labels_ == -1
    rows.append(
        {
            "detector": "DBSCAN",
            "settings": f"eps={eps:.3f}, min_samples={min_samples}",
            **_scores(y_outlier, db_flag),
        }
    )
    artefacts.update(dbscan=db, dbscan_flag=db_flag, eps_table=eps_table)

    iso = IsolationForest(contamination=contamination, random_state=random_state).fit(X)
    rows.append(
        {
            "detector": "Isolation Forest",
            "settings": "100 trees",
            **_scores(y_outlier, iso.predict(X) == -1, -iso.score_samples(X)),
        }
    )

    lof = LocalOutlierFactor(contamination=contamination)
    lof_flag = lof.fit_predict(X) == -1
    rows.append(
        {
            "detector": "Local Outlier Factor",
            "settings": "n_neighbors=20",
            **_scores(y_outlier, lof_flag, -lof.negative_outlier_factor_),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ee = EllipticEnvelope(contamination=contamination, random_state=random_state).fit(X)
    rows.append(
        {
            "detector": "Elliptic Envelope (Fast-MCD)",
            "settings": "single Gaussian",
            **_scores(y_outlier, ee.predict(X) == -1, -ee.score_samples(X)),
        }
    )

    return pd.DataFrame(rows), artefacts


def threshold_sweep(
    detector: GaussianMixtureAnomalyDetector, X, y_outlier, percentiles
) -> pd.DataFrame:
    """Precision / recall trade-off as the density threshold moves (book p. 266)."""
    dens = detector.score_samples(X)
    rows = []
    for p in percentiles:
        flag = dens < np.percentile(dens, p)
        rows.append({"percentile": p, **_scores(y_outlier, flag)})
    return pd.DataFrame(rows)


def evaluate_on_corruptions(
    detector: GaussianMixtureAnomalyDetector, X_clean: np.ndarray, random_state: int = 0
) -> tuple[pd.DataFrame, dict]:
    """False-positive rate on clean digits and detection rate per corruption type."""
    clean_flag = detector.predict(X_clean) == -1
    rows = [{"input": "clean test digits", "flagged_pct": 100 * float(clean_flag.mean())}]
    all_bad, all_scores = [], []
    for i, kind in enumerate(CORRUPTIONS):
        Xc = corrupt_digits(X_clean, kind, random_state + i)
        flag = detector.predict(Xc) == -1
        rows.append({"input": kind, "flagged_pct": 100 * float(flag.mean())})
        all_bad.append(Xc)
    X_bad = np.vstack(all_bad)
    y = np.r_[np.zeros(len(X_clean)), np.ones(len(X_bad))]
    all_scores = -detector.score_samples(np.vstack([X_clean, X_bad]))
    summary = {
        "clean_false_positive_pct": rows[0]["flagged_pct"],
        "corrupted_detection_pct": 100 * float(np.mean(detector.predict(X_bad) == -1)),
        "roc_auc_clean_vs_corrupted": float(roc_auc_score(y, all_scores)),
    }
    for r in rows[1:]:
        summary[f"detection_pct_{r['input']}"] = r["flagged_pct"]
    return pd.DataFrame(rows), summary


def error_rate_by_flag(classifier, detector, X, y) -> dict:
    """Does the detector flag the images the classifier gets wrong? (a proxy check on clean data)"""
    flagged = detector.predict(X) == -1
    wrong = classifier.predict(X) != y
    return {
        "classifier_error_pct_flagged": 100 * float(wrong[flagged].mean())
        if flagged.any()
        else float("nan"),
        "classifier_error_pct_not_flagged": 100 * float(wrong[~flagged].mean()),
        "n_flagged_test": int(flagged.sum()),
    }


@dataclass
class PartCResult:
    benchmark: pd.DataFrame  # every seed x dataset x detector
    benchmark_summary: pd.DataFrame  # mean / std over seeds
    sweeps: dict
    digits_detector: GaussianMixtureAnomalyDetector
    digits_ic_table: pd.DataFrame
    corruption_table: pd.DataFrame
    metrics: dict
    artefacts: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    run_id: str | None = None  # MLflow run that logged this part


def _slug(detector_name: str) -> str:
    return detector_name.split(" (")[0].lower().replace(" ", "_")


def run_synthetic_benchmark(
    cfg: dict, seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """GMM vs DBSCAN (+ reference detectors) on blobs and moons with injected outliers,
    repeated over several random datasets."""
    from .config import k_values as to_k
    from .data import make_blobs_with_outliers, make_moons_with_outliers

    c = float(cfg.get("contamination", 0.04))
    syn = cfg.get("synthetic", {})
    n = int(syn.get("n_samples", 1000))
    tables, sweeps, artefacts = [], {}, {}
    for i, s in enumerate(seeds):
        datasets = {
            "blobs": make_blobs_with_outliers(
                n, c, float(syn.get("blobs_min_mahalanobis", 3.0)), s
            ),
            "moons": make_moons_with_outliers(
                n, 0.05, c, float(syn.get("moons_min_distance", 0.15)), s
            ),
        }
        for name, (X, y) in datasets.items():
            table, art = benchmark_detectors(
                X,
                y,
                c,
                to_k(syn.get("gmm_k", {"start": 1, "stop": 11})),
                int(syn.get("n_init", 5)),
                int(syn.get("dbscan_min_samples", 5)),
                s,
            )
            table.insert(0, "dataset", name)
            table.insert(1, "seed", s)
            tables.append(table)
            if i == 0:  # keep the first seed's artefacts for figures
                sweeps[name] = threshold_sweep(
                    art["gmm"], X, y, cfg.get("threshold_sweep", [1, 2, 3, 4, 5, 6, 8, 10])
                )
                artefacts[name] = {"X": X, "y": y, **art}
    benchmark = pd.concat(tables, ignore_index=True)
    summary = (
        benchmark.groupby(["dataset", "detector"], sort=False)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )
    return benchmark, summary, sweeps, artefacts


def run_part_c(split, cfg: dict, seed: int = 42) -> PartCResult:
    from .config import k_values as to_k

    t0 = time.perf_counter()
    c = float(cfg.get("contamination", 0.04))
    seeds = list(cfg.get("synthetic", {}).get("seeds", [seed]))
    benchmark, summary, sweeps, artefacts = run_synthetic_benchmark(cfg, seeds)

    # ---- digits: the detector that ships with the API ---------------------------------
    dg = cfg.get("digits", {})
    pca_var = dg.get("pca_variance", 0.95)
    best, ic = select_gmm(
        split.X_train,
        to_k(dg.get("gmm_k", {"start": 1, "stop": 21})),
        dg.get("covariance_types", list(COVARIANCE_TYPES)),
        dg.get("criterion", "bic"),
        pca_var,
        int(dg.get("selection_n_init", 3)),
        seed,
    )
    common = dict(
        **best,
        threshold_percentile=100 * c,
        pca_variance=pca_var,
        n_init=int(dg.get("n_init", 10)),
        random_state=seed,
    )
    # book approach: threshold on the training densities
    book_detector = GaussianMixtureAnomalyDetector(**common).fit(split.X_train)
    _, book_summary = evaluate_on_corruptions(book_detector, split.X_test, seed)
    # deployed approach: threshold on held-out densities
    cal = float(dg.get("calibration_fraction", 0.25))
    detector = GaussianMixtureAnomalyDetector(**common, calibration_fraction=cal).fit(split.X_train)
    corruption_table, corr_summary = evaluate_on_corruptions(detector, split.X_test, seed)

    metrics = {}
    for _, r in summary.iterrows():
        key = f"{r.dataset}_{_slug(r.detector)}"
        metrics[f"{key}_f1_mean"] = float(r.f1_mean)
        metrics[f"{key}_f1_std"] = float(r.f1_std) if not pd.isna(r.f1_std) else 0.0
        if not pd.isna(r.roc_auc):
            metrics[f"{key}_roc_auc_mean"] = float(r.roc_auc)
    metrics.update({f"digits_{k}": v for k, v in corr_summary.items()})
    metrics["digits_train_threshold_clean_false_positive_pct"] = book_summary[
        "clean_false_positive_pct"
    ]
    metrics.update(
        {
            "digits_gmm_n_components": best["n_components"],
            "digits_pca_dims": int(detector.pca_.n_components_)
            if detector.pca_ is not None
            else 64,
            "synthetic_n_seeds": len(seeds),
            "runtime_s": time.perf_counter() - t0,
        }
    )
    params = {
        "contamination": c,
        "digits_covariance_type": best["covariance_type"],
        "digits_n_components": best["n_components"],
        "pca_variance": pca_var,
        "calibration_fraction": cal,
        "synthetic_seeds": ",".join(map(str, seeds)),
    }
    return PartCResult(
        benchmark, summary, sweeps, detector, ic, corruption_table, metrics, artefacts, params
    )
