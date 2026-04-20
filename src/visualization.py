import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from probing import build_layer_xy

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