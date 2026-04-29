from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from probing import build_layer_xy

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

# ks = [1,2,3,4,5,6,7,8,9,10]
# accs = [0.5361,0.5246,0.5243,0.5193,0.5277,0.5354,0.5395,0.5451,0.5603,0.5644]

# plt.plot(ks, accs, marker="o")
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Test Accuracy")
# plt.title("Tone Classification Accuracy vs PCA Dimensions")
# plt.show()

# x = np.arange(1, 11)

# pc1 = [0.36612303,0.36558769,0.35751794,0.33902804,0.31931131,
#        0.29901363,0.28592556,0.27528069,0.26857634,0.26173551]

# pc2 = [0.4757258,0.41605431,0.28526157,0.09030153,-0.08307697,
#        -0.21970625,-0.29826287,-0.33361554,-0.35250406,-0.36244095]

# plt.plot(x, pc1, marker="o", label="PC1")
# plt.plot(x, pc2, marker="o", label="PC2")
# plt.axhline(0, linestyle="--")
# plt.xlabel("Normalized Time Point")
# plt.ylabel("Loading")
# plt.title("PCA Loadings on F0 Contours")
# plt.legend()
# plt.show()

def plot_multi_layer_accuracy(df):
    plt.figure(figsize=(9, 5))

    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].sort_values("k")
        plt.plot(
            sub["k"],
            sub["accuracy"],
            marker="o",
            linewidth=2,
            label=f"Layer {layer}"
        )

    plt.xlabel("Number of PCA dimensions (k)")
    plt.ylabel("Test accuracy")
    plt.title("Tone classification accuracy vs PCA dimensions")
    plt.xscale("log")
    plt.xticks(
        [1,2,3,5,10,20,50,100,200,400,768],
        [1,2,3,5,10,20,50,100,200,400,768]
    )
    plt.ylim(0.20, 1.00)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_multi_layer_variance(df):
    plt.figure(figsize=(9, 5))

    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].sort_values("k")
        plt.plot(
            sub["k"],
            sub["explained_variance"],
            marker="o",
            linewidth=2,
            label=f"Layer {layer}"
        )

    plt.xlabel("Number of PCA dimensions (k)")
    plt.ylabel("Cumulative explained variance")
    plt.title("Explained variance vs PCA dimensions")
    plt.xscale("log")
    plt.xticks(
        [1,2,3,5,10,20,50,100,200,400,768],
        [1,2,3,5,10,20,50,100,200,400,768]
    )
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_most_compact_layer(df, target_acc=0.90):
    rows = []

    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].sort_values("k")

        reached = sub[sub["accuracy"] >= target_acc]

        if len(reached) == 0:
            rows.append({
                "layer": layer,
                "min_k_for_target": None,
                "best_accuracy": sub["accuracy"].max()
            })
        else:
            first = reached.iloc[0]
            rows.append({
                "layer": layer,
                "min_k_for_target": int(first["k"]),
                "best_accuracy": sub["accuracy"].max()
            })

    out = pd.DataFrame(rows)
    print(out.sort_values("min_k_for_target", na_position="last"))
    return out

def run_pca_visuals(multi_df):
    plot_multi_layer_accuracy(multi_df)
    plot_multi_layer_variance(multi_df)
    compact_df = find_most_compact_layer(multi_df, target_acc=0.90)



# PCA intervention visuals


# ----------------------------
# 1) Standard confusion matrix
# ----------------------------
def plot_confusion_matrix_basic(y_true, y_pred, labels=(1, 2, 3, 4), title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# --------------------------------
# 2) Row-normalized confusion matrix
# --------------------------------
def plot_confusion_matrix_normalized(y_true, y_pred, labels=(1, 2, 3, 4), title="Normalized Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(float)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # annotate
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------
# 3) Side-by-side raw count and normalized confusion matrix
# --------------------------------------------------------
def plot_confusion_matrix_both(y_true, y_pred, labels=(1, 2, 3, 4), prefix="Test"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # counts
    im1 = axes[0].imshow(cm)
    axes[0].set_title(f"{prefix} Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, f"{int(cm[i, j])}", ha="center", va="center")

    fig.colorbar(im1, ax=axes[0])

    # normalized
    im2 = axes[1].imshow(cm_norm)
    axes[1].set_title(f"{prefix} Confusion Matrix (Row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 4) Confusion matrix from your hidden-state classifier
# ----------------------------------------------------
def run_and_plot_confusion(clf, x_test, y_test, labels=(1, 2, 3, 4), prefix="Test"):
    y_pred = clf.predict(x_test)

    plot_confusion_matrix_basic(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        title=f"{prefix} Confusion Matrix"
    )

    plot_confusion_matrix_normalized(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        title=f"{prefix} Normalized Confusion Matrix"
    )

    return y_pred


# ----------------------------------------------------------------
# 5) Intervention transition heatmap: pred_before -> pred_after
# ----------------------------------------------------------------
def plot_transition_heatmap(df, pc_idx=None, delta=None, normalize=True, title=None):
    """
    df should be your intervention sweep dataframe, e.g. all_pc_df
    with columns:
      - pred_before
      - pred_after
      - pc_idx
      - delta
    """
    sub = df.copy()

    if pc_idx is not None:
        sub = sub[sub["pc_idx"] == pc_idx]
    if delta is not None:
        sub = sub[sub["delta"] == delta]

    tab = pd.crosstab(
        sub["pred_before"],
        sub["pred_after"],
        normalize="index" if normalize else False
    )

    # make sure all 4 tones appear
    labels = [1, 2, 3, 4]
    tab = tab.reindex(index=labels, columns=labels, fill_value=0)

    values = tab.values

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values)

    if title is None:
        mode = "Normalized" if normalize else "Counts"
        title = f"Transition Heatmap ({mode}) | PC={pc_idx}, delta={delta}"

    ax.set_title(title)
    ax.set_xlabel("Predicted After")
    ax.set_ylabel("Predicted Before")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return tab


# -------------------------------------------------------
# 6) True tone -> pred_after heatmap for interventions
# -------------------------------------------------------
def plot_true_to_after_heatmap(df, pc_idx=None, delta=None, normalize=True, title=None):
    """
    df should contain:
      - true_tone
      - pred_after
      - pc_idx
      - delta
    """
    sub = df.copy()

    if pc_idx is not None:
        sub = sub[sub["pc_idx"] == pc_idx]
    if delta is not None:
        sub = sub[sub["delta"] == delta]

    tab = pd.crosstab(
        sub["true_tone"],
        sub["pred_after"],
        normalize="index" if normalize else False
    )

    labels = [1, 2, 3, 4]
    tab = tab.reindex(index=labels, columns=labels, fill_value=0)

    values = tab.values

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values)

    if title is None:
        mode = "Normalized" if normalize else "Counts"
        title = f"True→After Heatmap ({mode}) | PC={pc_idx}, delta={delta}"

    ax.set_title(title)
    ax.set_xlabel("Predicted After")
    ax.set_ylabel("True Tone")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return tab


def run_pca_intervention_confusion_matrices(train_items, test_items, all_pc_df, layer_idx=6):
    iv_x_train, iv_y_train = build_layer_xy(train_items, layer_idx)
    iv_x_test, iv_y_test = build_layer_xy(test_items, layer_idx)

    clf = LogisticRegression(max_iter=3000)
    clf.fit(iv_x_train, iv_y_train)

    y_pred = clf.predict(iv_x_test)
    print("Accuracy used for confusion matrix:", accuracy_score(iv_y_test, y_pred))

    plot_confusion_matrix_basic(
        iv_y_test, y_pred,
        title=f"Layer {layer_idx} Raw Hidden-State Confusion Matrix"
    )
    plot_confusion_matrix_normalized(
        iv_y_test, y_pred,
        title=f"Layer {layer_idx} Raw Hidden-State Normalized Confusion Matrix"
    )
    plot_confusion_matrix_both(
        iv_y_test, y_pred,
        prefix=f"Layer {layer_idx} Raw Hidden-State"
    )

    plot_transition_heatmap(all_pc_df, pc_idx=4, delta=3, normalize=True)
    plot_true_to_after_heatmap(all_pc_df, pc_idx=4, delta=3, normalize=True)

    plot_transition_heatmap(all_pc_df, pc_idx=0, delta=3, normalize=True)
    plot_true_to_after_heatmap(all_pc_df, pc_idx=0, delta=3, normalize=True)

    plot_transition_heatmap(all_pc_df, pc_idx=4, delta=3, normalize=False)
    plot_true_to_after_heatmap(all_pc_df, pc_idx=4, delta=3, normalize=False)

# -------------------------
# General helpers
# -------------------------

def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Probing visualizations
# -------------------------

def save_probe_results(
    probe_results: List[Dict[str, Any]],
    out_path: str | Path = "./results/probing/layerwise_probe_results.csv",
) -> pd.DataFrame:
    """
    Save layerwise probe results.

    Expected probe_results:
        [
            {"layer": 0, "accuracy": 0.90},
            {"layer": 1, "accuracy": 0.91},
            ...
        ]
    """
    out_path = ensure_parent(out_path)

    df = pd.DataFrame(probe_results)
    df = df.sort_values("layer").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"[visualization] saved probe CSV: {out_path}")
    return df


def plot_layerwise_probe(
    probe_results: List[Dict[str, Any]] | pd.DataFrame,
    out_path: str | Path = "./results/probing/layerwise_probe_accuracy.png",
    title: str = "Layerwise Tone Probing Accuracy",
) -> pd.DataFrame:
    """
    Plot probe accuracy by wav2vec layer.
    """
    out_path = ensure_parent(out_path)

    if isinstance(probe_results, pd.DataFrame):
        df = probe_results.copy()
    else:
        df = pd.DataFrame(probe_results)

    df = df.sort_values("layer").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["layer"], df["accuracy"], marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(df["layer"].tolist())
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    best_idx = df["accuracy"].idxmax()
    best_layer = int(df.loc[best_idx, "layer"])
    best_acc = float(df.loc[best_idx, "accuracy"])

    ax.scatter([best_layer], [best_acc], s=80)
    ax.annotate(
        f"Best L{best_layer}\n{best_acc:.3f}",
        xy=(best_layer, best_acc),
        xytext=(best_layer, min(best_acc + 0.08, 0.98)),
        ha="center",
        arrowprops={"arrowstyle": "->"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[visualization] saved probe plot: {out_path}")
    return df


# -------------------------
# DAS result loading
# -------------------------

def das_metrics_to_df(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert DAS all_metrics list into a compact dataframe.
    """
    rows = []

    for m in metrics:
        rows.append(
            {
                "layer_idx": int(m["layer_idx"]),
                "k": int(m["k"]),
                "classifier_test_acc": float(m.get("classifier_test_acc", np.nan)),
                "target_success_rate": float(m["target_success_rate"]),
                "target_success_given_base_correct": float(
                    m.get("target_success_given_base_correct", np.nan)
                ),
                "flip_rate": float(m["flip_rate"]),
                "num_pairs": int(m["num_pairs"]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["layer_idx", "k"]).reset_index(drop=True)
    return df


def load_das_sweep(
    metrics_path: str | Path = "./results/das_layer_sweep/all_metrics.json",
) -> pd.DataFrame:
    metrics = load_json(metrics_path)
    return das_metrics_to_df(metrics)


def save_das_summary_csv(
    df: pd.DataFrame,
    out_path: str | Path = "./results/das_layer_sweep/das_summary.csv",
) -> None:
    out_path = ensure_parent(out_path)
    df.to_csv(out_path, index=False)
    print(f"[visualization] saved DAS summary CSV: {out_path}")

def plot_das_target_success_by_layer(
    df: pd.DataFrame,
    out_path: str | Path = "./results/das_layer_sweep/das_target_success_by_layer.png",
    title: str = "DAS Target Success by Layer",
) -> pd.DataFrame:
    """
    One line per k.
    x-axis: layer
    y-axis: target_success_rate
    """
    out_path = ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for k, sub in df.groupby("k"):
        sub = sub.sort_values("layer_idx")
        ax.plot(
            sub["layer_idx"],
            sub["target_success_rate"],
            marker="o",
            label=f"k={k}",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Target Success Rate")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sorted(df["layer_idx"].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="DAS dimensions")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[visualization] saved DAS layer plot: {out_path}")
    return df


def plot_das_vs_classifier_by_layer(
    df: pd.DataFrame,
    out_path: str | Path = "./results/das_layer_sweep/das_vs_classifier_by_layer.png",
    k_for_das: int = 4,
    title: str = "DAS Target Success vs. Classifier Accuracy",
) -> pd.DataFrame:
    """
    Compare ordinary classifier accuracy and DAS target success for one k.
    """
    out_path = ensure_parent(out_path)

    sub = df[df["k"] == k_for_das].copy()
    sub = sub.sort_values("layer_idx")

    if sub.empty:
        raise ValueError(f"No DAS results found for k={k_for_das}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        sub["layer_idx"],
        sub["classifier_test_acc"],
        marker="o",
        label="Classifier accuracy",
    )
    ax.plot(
        sub["layer_idx"],
        sub["target_success_rate"],
        marker="s",
        label=f"DAS target success, k={k_for_das}",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sub["layer_idx"].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[visualization] saved DAS-vs-classifier plot: {out_path}")
    return sub


def plot_das_k_saturation(
    df: pd.DataFrame,
    out_path: str | Path = "./results/das_layer_sweep/das_k_saturation.png",
    selected_layers: Sequence[int] | None = None,
    title: str = "DAS k-Saturation Curve",
) -> pd.DataFrame:
    """
    One line per layer.
    x-axis: k
    y-axis: target_success_rate
    """
    out_path = ensure_parent(out_path)

    plot_df = df.copy()

    if selected_layers is not None:
        plot_df = plot_df[plot_df["layer_idx"].isin(selected_layers)].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for layer_idx, sub in plot_df.groupby("layer_idx"):
        sub = sub.sort_values("k")
        ax.plot(
            sub["k"],
            sub["target_success_rate"],
            marker="o",
            label=f"L{layer_idx}",
        )

    ax.set_xlabel("DAS subspace size k")
    ax.set_ylabel("Target Success Rate")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sorted(plot_df["k"].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Layer", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[visualization] saved DAS k-saturation plot: {out_path}")
    return plot_df


def transition_metrics_to_matrix(
    transition_metrics: Dict[str, Dict[str, float]],
    value_key: str = "target_success_rate",
    n_classes: int = 4,
) -> np.ndarray:
    """
    Convert:
        {"0->1": {"target_success_rate": ...}, ...}

    into a 4x4 matrix.

    Diagonal is NaN because no same-tone interventions are used.
    """
    mat = np.full((n_classes, n_classes), np.nan)

    for transition, vals in transition_metrics.items():
        src, tgt = transition.split("->")
        src = int(src)
        tgt = int(tgt)
        mat[src, tgt] = float(vals[value_key])

    return mat


def plot_das_transition_heatmap_from_metrics(
    metrics: Dict[str, Any],
    out_path: str | Path,
    value_key: str = "target_success_rate",
    title: str | None = None,
    class_labels: Sequence[str] = ("T1", "T2", "T3", "T4"),
) -> np.ndarray:
    """
    Plot transition heatmap for a single DAS metrics object.
    Rows = base tone
    Columns = source/target tone
    """
    out_path = ensure_parent(out_path)

    mat = transition_metrics_to_matrix(
        metrics["transition_metrics"],
        value_key=value_key,
        n_classes=len(class_labels),
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(mat, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Source / target tone")
    ax.set_ylabel("Base tone")

    if title is None:
        layer = metrics.get("layer_idx", "?")
        k = metrics.get("k", "?")
        title = f"DAS Transition Success, Layer {layer}, k={k}"

    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                text = "-"
            else:
                text = f"{mat[i, j]:.2f}"

            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, label=value_key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[visualization] saved transition heatmap: {out_path}")
    return mat

def run_das_visuals(
    metrics_path: str | Path = "./results/das_layer_sweep/all_metrics.json",
    out_dir: str | Path = "./results/das_layer_sweep/figures",
    k_for_comparison: int = 4,
) -> pd.DataFrame:
    """
    Generate standard DAS visualizations from all_metrics.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_das_sweep(metrics_path)

    save_das_summary_csv(
        df,
        out_path=out_dir / "das_summary.csv",
    )

    plot_das_target_success_by_layer(
        df,
        out_path=out_dir / "das_target_success_by_layer.png",
    )

    plot_das_vs_classifier_by_layer(
        df,
        out_path=out_dir / "das_vs_classifier_by_layer.png",
        k_for_das=k_for_comparison,
    )

    plot_das_k_saturation(
        df,
        out_path=out_dir / "das_k_saturation_all_layers.png",
    )

    plot_das_k_saturation(
        df,
        out_path=out_dir / "das_k_saturation_selected_layers.png",
        selected_layers=[0, 2, 6, 8, 12],
    )

    # Transition heatmap for best layer/k if available.
    layer6_k4_path = Path(metrics_path).parent / "layer_6" / "k_4" / "metrics.json"

    if layer6_k4_path.exists():
        metrics = load_json(layer6_k4_path)
        plot_das_transition_heatmap_from_metrics(
            metrics,
            out_path=out_dir / "layer6_k4_transition_heatmap.png",
            title="Layer 6 DAS Transition Success, k=4",
        )
    else:
        print(f"[visualization] skipped heatmap; file not found: {layer6_k4_path}")

    return df