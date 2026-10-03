"""Part B - clustering for semi-supervised learning (Geron, Ch. 9, pp. 254-256).

Scenario: plenty of unlabelled images, budget to label only ``k = 50`` of them.

1. Baseline - label 50 *random* images.
2. Cluster the training set into 50 clusters and label only the image closest to each centroid
   (the *representative* images).
3. *Label propagation* - copy each representative's label to every image in its cluster.
4. *Partial propagation* - copy it only to the ``percentile`` % of images closest to the
   centroid, where propagated labels are most reliable.
5. (Extension from the book's "Active Learning" box) uncertainty sampling: ask the oracle
   for the images the model is least sure about, a few at a time.

Labels for step 2 come from an *oracle* (the ground truth, which simulates a human labeller)
or from a real human through the web UI's labelling page.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .clustering_features import make_logreg
from .data import DigitsSplit


def find_representatives(
    X_train: np.ndarray, k: int = 50, random_state: int = 42, n_init: int = 10
) -> tuple[KMeans, np.ndarray, np.ndarray]:
    """Cluster into ``k`` groups; return (kmeans, distance matrix, index of the image closest
    to each centroid)."""
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    distances = kmeans.fit_transform(X_train)
    representative_idx = np.argmin(distances, axis=0)
    return kmeans, distances, representative_idx


def propagate_labels(cluster_labels: np.ndarray, representative_labels: np.ndarray) -> np.ndarray:
    """Give every instance the label of its cluster's representative."""
    return np.asarray(representative_labels)[cluster_labels].astype(np.int32)


def partial_propagation_mask(
    distances: np.ndarray, cluster_labels: np.ndarray, percentile: float
) -> np.ndarray:
    """True for instances within the ``percentile``-th distance percentile of their own cluster."""
    own_dist = distances[np.arange(len(cluster_labels)), cluster_labels]
    mask = np.zeros(len(cluster_labels), dtype=bool)
    for c in np.unique(cluster_labels):
        in_cluster = cluster_labels == c
        cutoff = np.percentile(own_dist[in_cluster], percentile)
        mask |= in_cluster & (own_dist <= cutoff)
    return mask


def resolve_representative_labels(
    representative_idx: np.ndarray,
    y_oracle: np.ndarray,
    human_labels: Mapping[int, int] | None = None,
    allow_oracle_fallback: bool = False,
) -> tuple[np.ndarray, str]:
    """Labels for the representative images: from a human (via the UI) or from the oracle."""
    if not human_labels:
        return y_oracle[representative_idx], "oracle"
    labels, missing = [], []
    for idx in representative_idx:
        if int(idx) in human_labels:
            labels.append(int(human_labels[int(idx)]))
        elif allow_oracle_fallback:
            labels.append(int(y_oracle[idx]))
            missing.append(int(idx))
        else:
            missing.append(int(idx))
    if missing and not allow_oracle_fallback:
        raise ValueError(
            f"{len(missing)} representative images have no human label (e.g. {missing[:5]}). "
            "Label them in the web UI or set semi_supervised.allow_oracle_fallback: true."
        )
    return np.array(labels, dtype=int), "human" if not missing else "human+oracle"


def active_learning(
    X_pool: np.ndarray,
    y_oracle: np.ndarray,
    labeled_mask: np.ndarray,
    y_labeled: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_human_start: int,
    rounds: int = 5,
    queries_per_round: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Uncertainty sampling: each round, query the oracle for the ``queries_per_round`` pool
    instances with the lowest max class probability, add them, retrain."""
    labeled_mask = labeled_mask.copy()
    y_work = y_labeled.copy()
    rows = []
    n_human = n_human_start
    for r in range(rounds + 1):
        model = make_logreg(random_state).fit(X_pool[labeled_mask], y_work[labeled_mask])
        rows.append(
            {"round": r, "human_labels": n_human, "test_accuracy": model.score(X_test, y_test)}
        )
        if r == rounds:
            break
        unlabeled = np.flatnonzero(~labeled_mask)
        confidence = model.predict_proba(X_pool[unlabeled]).max(axis=1)
        query = unlabeled[np.argsort(confidence)[:queries_per_round]]
        y_work[query] = y_oracle[query]
        labeled_mask[query] = True
        n_human += len(query)
    return pd.DataFrame(rows)


def random_label_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    budgets: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    """Accuracy of Logistic Regression trained on ``n`` randomly chosen true labels."""
    rows = []
    for n in budgets:
        for s in seeds:
            rng = np.random.default_rng(s)
            idx = rng.choice(len(X_train), size=n, replace=False)
            if len(np.unique(y_train[idx])) < 2:
                continue
            acc = make_logreg(s).fit(X_train[idx], y_train[idx]).score(X_test, y_test)
            rows.append({"human_labels": n, "seed": s, "test_accuracy": acc})
    return (
        pd.DataFrame(rows)
        .groupby("human_labels", as_index=False)
        .test_accuracy.agg(["mean", "std"])
    )


def semi_supervised_robustness(
    seeds: list[int], k: int = 50, percentile: float = 20, n_init: int = 10, test_size: float = 0.25
) -> pd.DataFrame:
    """Random-50 vs representative-50 vs partial propagation on several random splits."""
    from .data import load_digits_split

    rows = []
    for s in seeds:
        sp = load_digits_split(test_size=test_size, random_state=s)
        Xtr, ytr, Xte, yte = sp.X_train, sp.y_train, sp.X_test, sp.y_test
        km, dist, rep = find_representatives(Xtr, k, s, n_init)
        y_prop = propagate_labels(km.labels_, ytr[rep])
        m = partial_propagation_mask(dist, km.labels_, percentile)
        m[rep] = True
        rows.append(
            {
                "seed": s,
                "random50_accuracy": make_logreg(s).fit(Xtr[:k], ytr[:k]).score(Xte, yte),
                "representative50_accuracy": make_logreg(s).fit(Xtr[rep], ytr[rep]).score(Xte, yte),
                "partial_propagation_accuracy": make_logreg(s)
                .fit(Xtr[m], y_prop[m])
                .score(Xte, yte),
                "partial_propagation_label_accuracy": float(np.mean(y_prop[m] == ytr[m])),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class PartBResult:
    model: LogisticRegression
    metrics: dict
    representatives: dict
    percentile_sweep: pd.DataFrame
    active_learning: pd.DataFrame | None
    random_curve: pd.DataFrame | None
    robustness: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)
    run_id: str | None = None  # MLflow run that logged this part


def run_part_b(
    split: DigitsSplit,
    cfg: dict,
    seed: int = 42,
    human_labels: Mapping[int, int] | None = None,
) -> PartBResult:
    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte = split.X_train, split.y_train, split.X_test, split.y_test
    k = int(cfg.get("n_labeled", 50))
    pct = float(cfg.get("percentile_closest", 20))
    n_init = int(cfg.get("n_init", 10))

    # 1. random-label baseline (the book uses the first 50 images of a shuffled split)
    random_model = make_logreg(seed).fit(Xtr[:k], ytr[:k])
    acc_random = random_model.score(Xte, yte)

    # 2. representative images
    kmeans, distances, rep_idx = find_representatives(Xtr, k, seed, n_init)
    y_rep, label_source = resolve_representative_labels(
        rep_idx, ytr, human_labels, bool(cfg.get("allow_oracle_fallback", False))
    )
    rep_model = make_logreg(seed).fit(Xtr[rep_idx], y_rep)
    acc_rep = rep_model.score(Xte, yte)

    # 3. full propagation
    y_prop = propagate_labels(kmeans.labels_, y_rep)
    acc_full = make_logreg(seed).fit(Xtr, y_prop).score(Xte, yte)

    # 4. partial propagation
    mask = partial_propagation_mask(distances, kmeans.labels_, pct)
    mask[rep_idx] = True  # representatives always keep their (human) label
    final_model = make_logreg(seed).fit(Xtr[mask], y_prop[mask])
    acc_partial = final_model.score(Xte, yte)

    sweep_rows = []
    for p in cfg.get("percentile_sweep", [5, 10, 20, 30, 50, 75, 100]):
        m = partial_propagation_mask(distances, kmeans.labels_, p)
        m[rep_idx] = True
        sweep_rows.append(
            {
                "percentile": p,
                "n_train": int(m.sum()),
                "label_accuracy": float(np.mean(y_prop[m] == ytr[m])),
                "test_accuracy": make_logreg(seed).fit(Xtr[m], y_prop[m]).score(Xte, yte),
            }
        )
    sweep = pd.DataFrame(sweep_rows)

    full_acc = make_logreg(seed).fit(Xtr, ytr).score(Xte, yte)
    metrics = {
        "n_human_labels": k,
        "label_fraction_pct": 100 * k / len(Xtr),
        "random50_accuracy": acc_random,
        "representative50_accuracy": acc_rep,
        "full_propagation_accuracy": acc_full,
        "full_propagation_label_accuracy": float(np.mean(y_prop == ytr)),
        "partial_propagation_accuracy": acc_partial,
        "partial_propagation_label_accuracy": float(np.mean(y_prop[mask] == ytr[mask])),
        "partial_propagation_n_train": int(mask.sum()),
        "fully_supervised_accuracy": full_acc,
        "representative_label_accuracy": float(np.mean(y_rep == ytr[rep_idx])),
    }

    al_df = rand_df = None
    al_cfg = cfg.get("active_learning", {})
    if al_cfg.get("enabled", False):
        al_df = active_learning(
            Xtr,
            ytr,
            mask,
            y_prop.copy(),
            Xte,
            yte,
            n_human_start=k,
            rounds=int(al_cfg.get("rounds", 5)),
            queries_per_round=int(al_cfg.get("queries_per_round", 10)),
            random_state=seed,
        )
        budgets = sorted(set(al_df.human_labels.tolist()))
        rand_df = random_label_curve(
            Xtr, ytr, Xte, yte, budgets, list(al_cfg.get("random_seeds", range(10)))
        )
        metrics["active_learning_final_accuracy"] = float(al_df.test_accuracy.iloc[-1])
        metrics["active_learning_final_labels"] = int(al_df.human_labels.iloc[-1])

    rob_df = None
    rob_cfg = cfg.get("robustness", {})
    if rob_cfg.get("enabled", False):
        rob_df = semi_supervised_robustness(list(rob_cfg["seeds"]), k, pct, n_init)
        for col in [
            "random50_accuracy",
            "partial_propagation_accuracy",
            "partial_propagation_label_accuracy",
        ]:
            metrics[f"robustness_{col}_mean"] = float(rob_df[col].mean())
            metrics[f"robustness_{col}_std"] = float(rob_df[col].std(ddof=1))
        metrics["robustness_n_splits"] = len(rob_df)

    metrics["runtime_s"] = time.perf_counter() - t0
    representatives = {
        "train_index": [int(i) for i in rep_idx],
        "pixels": [[float(v) for v in Xtr[i]] for i in rep_idx],
        "label_used": [int(v) for v in y_rep],
        "label_source": label_source,
        "split_seed": seed,
    }
    return PartBResult(
        model=final_model,
        metrics={
            key: (float(v) if isinstance(v, (np.floating, float)) else v)
            for key, v in metrics.items()
        },
        representatives=representatives,
        percentile_sweep=sweep,
        active_learning=al_df,
        random_curve=rand_df,
        robustness=rob_df,
        params={
            "n_labeled": k,
            "percentile_closest": pct,
            "label_source": label_source,
            "n_init": n_init,
        },
    )
