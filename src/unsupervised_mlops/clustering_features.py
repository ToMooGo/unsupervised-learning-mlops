"""Part A - K-Means as a feature-engineering step (Geron, Ch. 9, "Using Clustering for Preprocessing").

Each image is replaced by its distances to the ``k`` K-Means centroids, and a Logistic Regression
is trained on those distances. Because K-Means is only a preprocessing step, ``k`` is chosen
by the downstream classification accuracy under cross-validation (``GridSearchCV``), not by
inertia or silhouette score.

````scale_distances=True`` inserts a ``StandardScaler`` between K-Means and Logistic Regression.
The raw distances share a large common offset, which makes the optimisation ill-conditioned and
stops ``lbfgs`` from converging for some k; scaling (Geron, Ch. 2) fixes that. Both variants are
reported.

Two baselines are reported so the gain is not overstated: the book's Logistic Regression on raw
pixels, and a *strong* baseline - StandardScaler + Logistic Regression with ``C`` tuned by the same
cross-validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DigitsSplit, load_digits_split


def make_logreg(random_state: int = 42, max_iter: int = 10_000) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=random_state)


def build_kmeans_logreg_pipeline(
    n_clusters: int = 50,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 10_000,
    scale_distances: bool = False,
) -> Pipeline:
    """``KMeans -> [StandardScaler] -> LogisticRegression``: images become distances to
    ``n_clusters`` centroids."""
    steps = [("kmeans", KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state))]
    if scale_distances:
        steps.append(("scaler", StandardScaler()))
    steps.append(("log_reg", make_logreg(random_state, max_iter)))
    return Pipeline(steps)


DEFAULT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def tune_scaled_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: list[float] | None = None,
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
) -> GridSearchCV:
    """Strong baseline: StandardScaler -> LogisticRegression with C tuned by CV on training data."""
    pipe = Pipeline([("scaler", StandardScaler()), ("log_reg", make_logreg(random_state))])
    search = GridSearchCV(
        pipe, {"log_reg__C": list(c_grid or DEFAULT_C_GRID)}, cv=cv, n_jobs=n_jobs
    )
    return search.fit(X_train, y_train)


def logreg_converged(model) -> bool:
    """True if the (final) Logistic Regression stopped before ``max_iter``."""
    lr = model.named_steps["log_reg"] if isinstance(model, Pipeline) else model
    return bool(np.all(lr.n_iter_ < lr.max_iter))


def tune_n_clusters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
    n_init: int = 10,
    scale_distances: bool = False,
) -> GridSearchCV:
    """Grid-search ``k`` by cross-validated accuracy."""
    search = GridSearchCV(
        build_kmeans_logreg_pipeline(
            random_state=random_state, n_init=n_init, scale_distances=scale_distances
        ),
        param_grid={"kmeans__n_clusters": list(k_values)},
        cv=cv,
        n_jobs=n_jobs,
        scoring="accuracy",
    )
    return search.fit(X_train, y_train)


def clustering_diagnostics(
    X: np.ndarray, k_values: list[int], random_state: int = 42, n_init: int = 10
) -> pd.DataFrame:
    """Inertia and silhouette score per ``k`` (the *unsupervised* way to choose k)."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X, km.labels_)) if k > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def robustness_check(
    seeds: list[int],
    k_values: list[int],
    test_size: float = 0.25,
    cv: int = 3,
    n_jobs: int = -1,
    n_init: int = 10,
    scale_distances: bool = False,
) -> pd.DataFrame:
    """Repeat baseline vs. tuned pipeline on several random train/test splits.

    ``k`` is re-tuned inside every split (on that split's training set only), so no test
    information leaks into model selection.
    """
    rows = []
    for seed in seeds:
        split = load_digits_split(test_size=test_size, random_state=seed)
        base = make_logreg(seed).fit(split.X_train, split.y_train)
        strong = tune_scaled_baseline(
            split.X_train, split.y_train, cv=cv, n_jobs=n_jobs, random_state=seed
        )
        search = tune_n_clusters(
            split.X_train,
            split.y_train,
            k_values,
            cv=cv,
            n_jobs=n_jobs,
            random_state=seed,
            n_init=n_init,
            scale_distances=scale_distances,
        )
        rows.append(
            {
                "seed": seed,
                "baseline_accuracy": base.score(split.X_test, split.y_test),
                "tuned_baseline_accuracy": strong.score(split.X_test, split.y_test),
                "kmeans_logreg_accuracy": search.score(split.X_test, split.y_test),
                "best_k": int(search.best_params_["kmeans__n_clusters"]),
            }
        )
    df = pd.DataFrame(rows)
    df["gain_pp"] = 100 * (df.kmeans_logreg_accuracy - df.baseline_accuracy)
    df["gain_vs_tuned_pp"] = 100 * (df.kmeans_logreg_accuracy - df.tuned_baseline_accuracy)
    return df


@dataclass
class PartAResult:
    model: Pipeline
    metrics: dict
    grid_scores: pd.DataFrame
    diagnostics: pd.DataFrame
    robustness: pd.DataFrame | None = None
    ablation: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)
    run_id: str | None = None  # MLflow run that logged this part


def run_part_a(split: DigitsSplit, cfg: dict, seed: int = 42, n_jobs: int = -1) -> PartAResult:
    """Baseline -> fixed-k pipeline -> GridSearchCV over k -> multi-split check -> ablation."""
    from .config import k_values as to_k

    n_init = int(cfg.get("n_init", 10))
    cv = int(cfg.get("cv", 3))
    scale = bool(cfg.get("scale_distances", True))
    t0 = time.perf_counter()

    baseline = make_logreg(seed).fit(split.X_train, split.y_train)
    baseline_acc = baseline.score(split.X_test, split.y_test)
    baseline_cv = cross_val_score(make_logreg(seed), split.X_train, split.y_train, cv=cv).mean()
    strong = tune_scaled_baseline(
        split.X_train, split.y_train, cv=cv, n_jobs=n_jobs, random_state=seed
    )
    strong_acc = strong.score(split.X_test, split.y_test)

    fixed_k = int(cfg.get("initial_k", 50))
    fixed = build_kmeans_logreg_pipeline(fixed_k, seed, n_init, scale_distances=scale).fit(
        split.X_train, split.y_train
    )
    fixed_acc = fixed.score(split.X_test, split.y_test)

    grid = to_k(cfg["k_grid"])
    search = tune_n_clusters(
        split.X_train,
        split.y_train,
        grid,
        cv=cv,
        n_jobs=n_jobs,
        random_state=seed,
        n_init=n_init,
        scale_distances=scale,
    )
    best_k = int(search.best_params_["kmeans__n_clusters"])
    tuned_acc = search.score(split.X_test, split.y_test)

    grid_scores = pd.DataFrame(
        {
            "k": search.cv_results_["param_kmeans__n_clusters"].astype(int),
            "cv_accuracy": search.cv_results_["mean_test_score"],
            "cv_std": search.cv_results_["std_test_score"],
        }
    )
    diagnostics = clustering_diagnostics(split.X_train, to_k(cfg["diagnostics_k"]), seed, n_init)

    metrics = {
        "baseline_accuracy": baseline_acc,
        "baseline_cv_accuracy": float(baseline_cv),
        "initial_k_accuracy": fixed_acc,
        "tuned_accuracy": tuned_acc,
        "best_k": best_k,
        "best_cv_accuracy": float(search.best_score_),
        "tuned_baseline_accuracy": strong_acc,
        "tuned_baseline_cv_accuracy": float(strong.best_score_),
        "tuned_baseline_C": float(strong.best_params_["log_reg__C"]),
        "baseline_errors": int(round((1 - baseline_acc) * len(split.y_test))),
        "tuned_baseline_errors": int(round((1 - strong_acc) * len(split.y_test))),
        "tuned_errors": int(round((1 - tuned_acc) * len(split.y_test))),
        "n_test": len(split.y_test),
        "final_logreg_converged": logreg_converged(search.best_estimator_),
    }

    robustness = None
    rob_cfg = cfg.get("robustness", {})
    if rob_cfg.get("enabled", False):
        robustness = robustness_check(
            seeds=list(rob_cfg["seeds"]),
            k_values=to_k(rob_cfg["k_grid"]),
            cv=cv,
            n_jobs=n_jobs,
            n_init=n_init,
            scale_distances=scale,
        )
        metrics.update(
            {
                "robustness_baseline_mean": float(robustness.baseline_accuracy.mean()),
                "robustness_baseline_std": float(robustness.baseline_accuracy.std(ddof=1)),
                "robustness_tuned_mean": float(robustness.kmeans_logreg_accuracy.mean()),
                "robustness_tuned_std": float(robustness.kmeans_logreg_accuracy.std(ddof=1)),
                "robustness_tuned_baseline_mean": float(robustness.tuned_baseline_accuracy.mean()),
                "robustness_tuned_baseline_std": float(
                    robustness.tuned_baseline_accuracy.std(ddof=1)
                ),
                "robustness_gain_pp_mean": float(robustness.gain_pp.mean()),
                "robustness_gain_vs_tuned_pp_mean": float(robustness.gain_vs_tuned_pp.mean()),
                "robustness_splits_improved": int((robustness.gain_pp > 0).sum()),
                "robustness_splits_improved_vs_tuned": int((robustness.gain_vs_tuned_pp > 0).sum()),
                "robustness_n_splits": len(robustness),
            }
        )

    ablation = None
    abl_cfg = cfg.get("ablation", {})
    if abl_cfg.get("enabled", False):
        # identical procedure and k grid, distance features scaled vs. raw (the book's pipeline)
        rows = []
        for variant_scale in (True, False):
            s_ = tune_n_clusters(
                split.X_train,
                split.y_train,
                to_k(abl_cfg["k_grid"]),
                cv=cv,
                n_jobs=n_jobs,
                random_state=seed,
                n_init=n_init,
                scale_distances=variant_scale,
            )
            rows.append(
                {
                    "variant": "with StandardScaler" if variant_scale else "raw distances (book)",
                    "best_k": int(s_.best_params_["kmeans__n_clusters"]),
                    "cv_accuracy": float(s_.best_score_),
                    "test_accuracy": s_.score(split.X_test, split.y_test),
                    "fit_time_s": float(np.sum(s_.cv_results_["mean_fit_time"]) * cv),
                }
            )
        ablation = pd.DataFrame(rows)

    metrics["runtime_s"] = time.perf_counter() - t0
    return PartAResult(
        model=search.best_estimator_,
        metrics={
            k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()
        },
        grid_scores=grid_scores,
        diagnostics=diagnostics,
        robustness=robustness,
        ablation=ablation,
        params={
            "initial_k": fixed_k,
            "k_grid": f"{grid[0]}..{grid[-1]} ({len(grid)} values)",
            "cv": cv,
            "n_init": n_init,
            "scale_distances": scale,
            "robustness_k_grid": (
                ", ".join(map(str, to_k(rob_cfg["k_grid"]))) if robustness is not None else "n/a"
            ),
        },
    )
