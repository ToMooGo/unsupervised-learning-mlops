"""Figures for the report and README (matplotlib, saved as PNG).

Colour roles are fixed so the same entity always has the same colour across figures:
blue = K-Means / GMM (the project's main method), orange = DBSCAN or the comparison method,
aqua = a third series, neutral greys = reference / context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
NEUTRAL = "#b9b8b2"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _digit(ax, pixels, title=None, color=INK_2):
    ax.imshow(
        np.asarray(pixels).reshape(8, 8), cmap="binary", vmin=0, vmax=16, interpolation="nearest"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=8, color=color, loc="center", fontweight="normal", pad=2)


# --------------------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------------------
def plot_k_selection(
    grid: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best_k: int,
    baseline_cv: float | None,
    path,
    tuned_baseline_cv: float | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    g = grid.sort_values("k")
    ax.fill_between(
        g.k, g.cv_accuracy - g.cv_std, g.cv_accuracy + g.cv_std, color=BLUE, alpha=0.12, lw=0
    )
    ax.plot(g.k, g.cv_accuracy, color=BLUE, label="K-Means -> Scaler -> LogReg (3-fold CV)")
    if baseline_cv is not None:
        ax.axhline(baseline_cv, color=INK_2, lw=1.2, ls="--", label="LogReg on raw pixels (CV)")
    if tuned_baseline_cv is not None:
        ax.axhline(
            tuned_baseline_cv, color=ORANGE, lw=1.4, ls=":", label="Scaler + LogReg, C tuned (CV)"
        )
    best = g.loc[g.k == best_k].iloc[0]
    ax.scatter(
        [best_k], [best.cv_accuracy], s=60, color=BLUE, edgecolor="white", linewidth=2, zorder=5
    )
    ax.annotate(
        f"best k = {best_k}\nCV acc = {best.cv_accuracy:.3f}",
        (best_k, best.cv_accuracy),
        xytext=(-95, -38),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    floor = 0.90
    if g.cv_accuracy.min() < floor:  # zoom on the interesting range; tiny k is far below
        ax.set_ylim(
            floor, max(g.cv_accuracy.max(), baseline_cv or 0, tuned_baseline_cv or 0) + 0.006
        )
        k_low = int(g.loc[g.cv_accuracy < floor, "k"].max())
        ax.text(
            0.01,
            0.02,
            f"k <= {k_low}: CV accuracy below {floor:.2f} (off-scale)",
            transform=ax.transAxes,
            fontsize=8,
            color=INK_2,
        )
    ax.set(
        title="Choose k by downstream accuracy",
        xlabel="number of clusters k",
        ylabel="cross-validated accuracy",
    )
    ax.legend(loc="lower right")

    d = diagnostics.sort_values("k")
    axes[1].plot(d.k, d.inertia, color=NEUTRAL, marker="o", ms=4)
    axes[1].set(title="Inertia (elbow)", xlabel="k", ylabel="inertia")
    axes[2].plot(d.k, d.silhouette, color=NEUTRAL, marker="o", ms=4)
    axes[2].set(title="Silhouette score", xlabel="k", ylabel="silhouette")
    fig.suptitle(
        "Unsupervised criteria (grey) favour small k; the classifier prefers many more clusters",
        x=0.01,
        ha="left",
        fontsize=9.5,
        color=INK_2,
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_robustness(
    rob: pd.DataFrame,
    path,
    left="baseline_accuracy",
    right="kmeans_logreg_accuracy",
    left_label="LogReg (raw pixels)",
    right_label="K-Means -> LogReg",
    title=None,
):
    rob = rob.sort_values("seed").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rob) + 1.4))
    y = np.arange(len(rob))
    for i, r in rob.iterrows():
        ax.plot([r[left] * 100, r[right] * 100], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter(
        rob[left] * 100,
        y,
        color=NEUTRAL,
        s=55,
        zorder=3,
        label=left_label,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.scatter(
        rob[right] * 100,
        y,
        color=BLUE,
        s=55,
        zorder=4,
        label=right_label,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_yticks(y, [f"split seed {s}" for s in rob.seed])
    ax.set_xlabel("test accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title or "Gain holds across random train/test splits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_robustness_a(rob: pd.DataFrame, path):
    """Per-split test accuracy of both baselines and the K-Means pipeline."""
    rob = rob.sort_values("seed").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(rob) + 1.6))
    y = np.arange(len(rob))
    series = [
        ("baseline_accuracy", "LogReg, raw pixels (book)", NEUTRAL),
        ("tuned_baseline_accuracy", "Scaler + LogReg, C tuned", ORANGE),
        ("kmeans_logreg_accuracy", "K-Means -> Scaler -> LogReg", BLUE),
    ]
    lo = rob[[c for c, _, _ in series]].min(axis=1) * 100
    hi = rob[[c for c, _, _ in series]].max(axis=1) * 100
    ax.hlines(y, lo, hi, color=GRID, lw=3, zorder=1)
    for col, label, color in series:
        ax.scatter(
            rob[col] * 100,
            y,
            color=color,
            s=55,
            zorder=3,
            label=label,
            edgecolor="white",
            linewidth=1.5,
        )
    ax.set_yticks(y, [f"split seed {s}" for s in rob.seed])
    ax.set_xlabel("test accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Test accuracy on five random splits (all hyperparameters tuned per split)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------------------
# Part B
# --------------------------------------------------------------------------------------
def plot_representatives(reps: dict, path, y_true=None):
    pixels, labels = reps["pixels"], reps["label_used"]
    n = len(pixels)
    cols = 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 1.05))
    for i, ax in enumerate(np.ravel(axes)):
        if i >= n:
            ax.axis("off")
            continue
        wrong = y_true is not None and y_true[i] != labels[i]
        _digit(
            ax, pixels[i], f"{labels[i]}" + (" (!)" if wrong else ""), ORANGE if wrong else INK_2
        )
    fig.suptitle(
        f"The {n} cluster-representative images - the only images a human labels",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_label_efficiency(
    metrics: dict, random_curve: pd.DataFrame | None, al: pd.DataFrame | None, n_train: int, path
):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    if random_curve is not None and len(random_curve):
        rc = random_curve.sort_values("human_labels")
        ax.fill_between(
            rc.human_labels,
            100 * (rc["mean"] - rc["std"]),
            100 * (rc["mean"] + rc["std"]),
            color=NEUTRAL,
            alpha=0.25,
            lw=0,
        )
        ax.plot(
            rc.human_labels,
            100 * rc["mean"],
            color=MUTED,
            marker="o",
            ms=5,
            label="random images labelled (mean ± std, 10 draws)",
        )
    if al is not None and len(al):
        ax.plot(
            al.human_labels,
            100 * al.test_accuracy,
            color=AQUA,
            marker="o",
            ms=5,
            label="+ active learning (uncertainty sampling)",
        )
    k = metrics["n_human_labels"]
    pts = [
        (metrics["random50_accuracy"], "random 50", NEUTRAL),
        (metrics["representative50_accuracy"], "50 representatives", ORANGE),
        (
            metrics["partial_propagation_accuracy"],
            "50 representatives\n+ partial propagation",
            BLUE,
        ),
    ]
    for acc, lab, col in pts:
        ax.scatter([k], [100 * acc], s=70, color=col, edgecolor="white", linewidth=1.8, zorder=5)
        ax.annotate(
            f"{lab}: {100 * acc:.1f}%",
            (k, 100 * acc),
            xytext=(10, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            va="center",
        )
    full = 100 * metrics["fully_supervised_accuracy"]
    x_right = max(al.human_labels.max() if al is not None else k, k) * 1.02
    ax.axhline(full, color=INK_2, ls="--", lw=1.2)
    ax.text(
        x_right,
        full + 0.4,
        f"all {n_train} labels: {full:.1f}%",
        ha="right",
        fontsize=8.5,
        color=INK_2,
    )
    ax.set(
        xlabel="number of human-provided labels",
        ylabel="test accuracy (%)",
        title="Label efficiency: accuracy per human label",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_percentile_sweep(sweep: pd.DataFrame, chosen: float, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col, title in [
        (axes[0], "label_accuracy", "Accuracy of propagated labels"),
        (axes[1], "test_accuracy", "Test accuracy of the model"),
    ]:
        ax.plot(sweep.percentile, 100 * sweep[col], color=BLUE, marker="o", ms=5)
        sel = sweep.loc[sweep.percentile == chosen]
        if len(sel):
            ax.scatter(
                sel.percentile,
                100 * sel[col],
                s=90,
                facecolor="none",
                edgecolor=ORANGE,
                linewidth=2,
                zorder=5,
            )
            ax.annotate(
                f"chosen: {chosen:g}% -> {100 * sel[col].iloc[0]:.1f}%",
                (chosen, 100 * sel[col].iloc[0]),
                xytext=(12, 10),
                textcoords="offset points",
                fontsize=8.5,
                color=INK,
            )
        ax.set(title=title, xlabel="propagate to the closest x% of each cluster", ylabel="%")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------------------
# Part C
# --------------------------------------------------------------------------------------
def _density_contours(ax, detector, X):
    pad = 0.15 * (X.max(0) - X.min(0))
    x0, y0 = X.min(0) - pad
    x1, y1 = X.max(0) + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250))
    zz = detector.score_samples(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    levels = np.linspace(max(zz.min(), detector.threshold_ - 6), zz.max(), 8)
    ax.contourf(xx, yy, zz, levels=levels, cmap="Blues", alpha=0.35)
    ax.contour(
        xx, yy, zz, levels=[detector.threshold_], colors=[BLUE], linewidths=1.6, linestyles="--"
    )


def plot_synthetic_detection(artefacts: dict, summary: pd.DataFrame, path):
    names = list(artefacts)
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 4.6 * len(names)))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        a = artefacts[name]
        X, y = a["X"], a["y"]
        for c, (method, flag_key) in enumerate(
            [("GMM (BIC-selected)", "gmm_flag"), ("DBSCAN", "dbscan_flag")]
        ):
            ax = axes[r, c]
            flag = a[flag_key]
            if c == 0:
                _density_contours(ax, a["gmm"], X)
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=6, color=NEUTRAL, lw=0, label="inlier")
            ax.scatter(
                X[y == 1, 0],
                X[y == 1, 1],
                s=46,
                facecolor="none",
                edgecolor=INK,
                lw=1.1,
                label="injected outlier (truth)",
            )
            col = BLUE if c == 0 else ORANGE
            ax.scatter(
                X[flag, 0],
                X[flag, 1],
                s=16,
                marker="x",
                color=col,
                lw=1.6,
                label=f"flagged by {method.split(' ')[0]}",
            )
            row = summary[(summary.dataset == name) & (summary.detector == method)].iloc[0]
            spread = "" if pd.isna(row.f1_std) else f" ± {row.f1_std:.2f}"
            ax.set_title(f"{name}: {method} - F1 {row.f1_mean:.2f}{spread}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_benchmark_f1(summary: pd.DataFrame, n_seeds: int, path):
    datasets = list(dict.fromkeys(summary.dataset))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"GMM (BIC-selected)": BLUE, "DBSCAN": ORANGE}
    for ax, ds in zip(axes, datasets, strict=True):
        s = summary[summary.dataset == ds].iloc[::-1]
        y = np.arange(len(s))
        ax.barh(
            y,
            s.f1_mean,
            xerr=s.f1_std.fillna(0),
            height=0.62,
            color=[colors.get(d, NEUTRAL) for d in s.detector],
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2},
        )
        for yi, v in zip(y, s.f1_mean, strict=True):
            ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, s.detector)
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set(
            title=f"{ds} + 4% injected outliers", xlabel=f"F1 (mean ± std over {n_seeds} datasets)"
        )
    fig.tight_layout()
    return _save(fig, path)


def plot_information_criteria(
    ic_blobs: pd.DataFrame, ic_digits: pd.DataFrame, best_digits: dict, path
):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    ax = axes[0]
    f = ic_blobs[ic_blobs.covariance_type == "full"].sort_values("k")
    ax.plot(f.k, f.bic, color=BLUE, marker="o", ms=4, label="BIC")
    ax.plot(f.k, f.aic, color=ORANGE, marker="o", ms=4, label="AIC")
    ax.set(
        title="Blobs: BIC / AIC vs k (full covariance)",
        xlabel="number of components k",
        ylabel="criterion (lower is better)",
    )
    ax.legend()
    ax = axes[1]
    palette = {"full": BLUE, "tied": ORANGE, "diag": AQUA, "spherical": VIOLET}
    for cov, grp in ic_digits.groupby("covariance_type"):
        g = grp.sort_values("k")
        ax.plot(g.k, g.bic, color=palette[cov], marker="o", ms=3, lw=1.6, label=cov)
    b = ic_digits[
        (ic_digits.k == best_digits["n_components"])
        & (ic_digits.covariance_type == best_digits["covariance_type"])
    ]
    ax.scatter(b.k, b.bic, s=110, facecolor="none", edgecolor=INK, lw=1.6, zorder=6)
    ax.annotate(
        f"selected: k={best_digits['n_components']}, {best_digits['covariance_type']}",
        (b.k.iloc[0], b.bic.iloc[0]),
        xytext=(14, 18),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
    )
    ax.set(
        title="Digits (PCA 95%): BIC by covariance type",
        xlabel="number of components k",
        ylabel="BIC",
    )
    ax.legend(title="covariance_type", title_fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, dataset: str, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(sweep.percentile, sweep.precision, color=BLUE, marker="o", ms=4, label="precision")
    ax.plot(sweep.percentile, sweep.recall, color=ORANGE, marker="o", ms=4, label="recall")
    ax.axvline(chosen, color=INK_2, ls="--", lw=1.1)
    ax.text(
        chosen + 0.15,
        0.05,
        f"{chosen:g}th percentile\n(true contamination)",
        fontsize=8,
        color=INK_2,
    )
    ax.set(
        ylim=(0, 1.05),
        xlabel="density threshold percentile",
        ylabel="score",
        title=f"Precision / recall trade-off of the threshold ({dataset})",
    )
    ax.legend(loc="center right")
    fig.tight_layout()
    return _save(fig, path)


def plot_digit_anomalies(detector, X_train, path, n: int = 20):
    dens = detector.score_samples(X_train)
    idx = np.argsort(dens)[:n]
    cols = 10
    fig, axes = plt.subplots(
        int(np.ceil(n / cols)), cols, figsize=(cols * 0.95, 1.15 * int(np.ceil(n / cols)))
    )
    for ax, i in zip(np.ravel(axes), idx, strict=False):
        _digit(ax, X_train[i], f"log p={dens[i]:.0f}")
    fig.suptitle(
        f"The {n} lowest-density training digits (the detector's idea of 'unusual handwriting')",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_corruptions(
    table: pd.DataFrame, examples: dict, target_pct: float, descriptions: dict, path
):
    t = table.copy()
    t["label"] = [
        ("clean test digits" if r == "clean test digits" else f"{r} - {descriptions.get(r, '')}")
        for r in t.input
    ]
    t = t.iloc[::-1].reset_index(drop=True)
    fig = plt.figure(figsize=(11.5, 4.9))
    gs = fig.add_gridspec(2, len(examples), height_ratios=[1, 3.3], hspace=0.45)
    for j, (kind, px) in enumerate(examples.items()):
        _digit(fig.add_subplot(gs[0, j]), px, kind)
    ax = fig.add_subplot(gs[1, :])
    colors = [NEUTRAL if r == "clean test digits" else BLUE for r in t.input]
    ax.barh(np.arange(len(t)), t.flagged_pct, color=colors, height=0.62)
    for yi, v in enumerate(t.flagged_pct):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
    ax.axvline(target_pct, color=ORANGE, ls="--", lw=1.3)
    ax.text(
        target_pct + 0.8,
        len(t) - 0.45,
        f"target false-positive rate {target_pct:g}%",
        color=INK_2,
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(t)), t.label)
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    ax.set(
        xlabel="% of test images flagged as anomalous",
        title="Digit anomaly detector: false alarms on clean data vs detection of corrupted inputs",
    )
    return _save(fig, path)
