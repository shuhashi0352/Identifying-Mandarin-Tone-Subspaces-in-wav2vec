import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from probing import build_layer_xy


# -------------------------------------------------
# 1) Build supervised tone subspace from probe weights
# -------------------------------------------------
def fit_supervised_subspace(train_items, test_items, layer_idx=6, subspace_dim=5):
    """
    Fit multinomial logistic regression on raw hidden states.
    Use the classifier weight matrix to derive a supervised tone subspace.

    Returns a bundle parallel to the PCA bundle.
    """
    x_train, y_train = build_layer_xy(train_items, layer_idx)
    x_test, y_test = build_layer_xy(test_items, layer_idx)

    probe = LogisticRegression(max_iter=3000)
    probe.fit(x_train, y_train)

    y_pred = probe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Layer {layer_idx} | supervised subspace dim = {subspace_dim}")
    print("Probe accuracy on original hidden states:", acc)
    print(classification_report(y_test, y_pred))

    # probe.coef_: [n_classes, D]
    W = probe.coef_

    # SVD of classifier weights
    # right singular vectors give discriminative directions in hidden space
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    # top-k supervised directions: [k, D]
    basis = Vt[:subspace_dim]

    return {
        "layer_idx": layer_idx,
        "subspace_dim": subspace_dim,
        "probe": probe,
        "basis": basis,          # [k, D]
        "singular_values": S,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_items": train_items,
        "test_items": test_items,
    }


# -------------------------------------------------
# 2) Project / reconstruct in supervised subspace
# -------------------------------------------------
def project_to_basis(x, basis):
    """
    x: [D]
    basis: [k, D]  (rows are orthonormal directions)
    returns z: [k]
    """
    return basis @ x


def reconstruct_from_basis(z, basis):
    """
    z: [k]
    basis: [k, D]
    returns x_rec: [D]
    """
    return basis.T @ z


def intervene_supervised_coordinates(x, basis, coord_updates):
    """
    Intervene on selected supervised-subspace coordinates.

    This operates only within the supervised low-rank subspace.
    """
    z_orig = project_to_basis(x, basis)
    z_mod = z_orig.copy()

    for idx, delta in coord_updates.items():
        z_mod[idx] += delta

    x_rec = reconstruct_from_basis(z_mod, basis)
    return z_orig, z_mod, x_rec


# -------------------------------------------------
# 3) Prediction helper
# -------------------------------------------------
def predict_probe(probe, x):
    pred = int(probe.predict(x.reshape(1, -1))[0])
    probs = probe.predict_proba(x.reshape(1, -1))[0]
    return pred, probs


# -------------------------------------------------
# 4) Single intervention for supervised subspace
# -------------------------------------------------
def run_single_supervised_intervention(bundle, test_index=0, coord_updates=None):
    """
    Run one supervised-subspace intervention on one test item.
    """
    if coord_updates is None:
        coord_updates = {0: 2.0}

    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    probe = bundle["probe"]
    basis = bundle["basis"]
    test_items = bundle["test_items"]

    x = x_test[test_index]
    true_tone = int(y_test[test_index])
    item = test_items[test_index]

    pred_before, probs_before = predict_probe(probe, x)
    z_orig, z_mod, x_mod = intervene_supervised_coordinates(x, basis, coord_updates)
    pred_after, probs_after = predict_probe(probe, x_mod)

    result = {
        "wav_file": item["wav_file"].name,
        "speaker": item["speaker"],
        "sound": item["sound"],
        "true_tone": true_tone,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "flipped": int(pred_before != pred_after),
        "coord_updates": coord_updates,
        "probs_before": probs_before,
        "probs_after": probs_after,
        "z_orig": z_orig,
        "z_mod": z_mod,
    }

    print("wav_file:", result["wav_file"])
    print("speaker:", result["speaker"])
    print("sound:", result["sound"])
    print("true tone:", result["true_tone"])
    print("pred before:", result["pred_before"])
    print("pred after :", result["pred_after"])
    print("flipped:", bool(result["flipped"]))
    print("coord updates:", result["coord_updates"])
    print("probs before:", np.round(result["probs_before"], 4))
    print("probs after :", np.round(result["probs_after"], 4))

    return result


# -------------------------------------------------
# 5) Sweep one supervised coordinate
# -------------------------------------------------
def sweep_single_supervised_coord(bundle, coord_idx=0, deltas=(-3, -2, -1, 1, 2, 3), max_items=500):
    """
    Sweep interventions on one supervised-subspace coordinate.
    """
    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    probe = bundle["probe"]
    basis = bundle["basis"]
    test_items = bundle["test_items"]

    n = min(max_items, len(x_test))
    rows = []

    for i in range(n):
        x = x_test[i]
        true_tone = int(y_test[i])
        item = test_items[i]

        pred_before, probs_before = predict_probe(probe, x)

        for delta in deltas:
            z_orig, z_mod, x_mod = intervene_supervised_coordinates(x, basis, {coord_idx: delta})
            pred_after, probs_after = predict_probe(probe, x_mod)

            rows.append({
                "test_index": i,
                "wav_file": item["wav_file"].name,
                "speaker": item["speaker"],
                "sound": item["sound"],
                "true_tone": true_tone,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "coord_idx": coord_idx,
                "delta": delta,
                "flipped": int(pred_before != pred_after),
                "correct_before": int(pred_before == true_tone),
                "correct_after": int(pred_after == true_tone),
                "prob_true_before": float(probs_before[true_tone - 1]),
                "prob_true_after": float(probs_after[true_tone - 1]),
            })

    df = pd.DataFrame(rows)

    print(f"\nSupervised coord {coord_idx+1} intervention summary")
    print(df.groupby("delta")[["flipped", "correct_before", "correct_after"]].mean())

    return df


# -------------------------------------------------
# 6) Compare PCA and supervised subspace at matched k
# -------------------------------------------------
def compare_pca_vs_supervised(
    train_items,
    test_items,
    layer_idx=6,
    subspace_dims=(1, 2, 3, 5),
    deltas=(-3, -2, -1, 1, 2, 3),
    max_items=500,
):
    """
    For each subspace dimension k:
      - build PCA bundle
      - build supervised bundle
      - sweep each coordinate
      - aggregate intervention stats

    Returns:
      comparison_df
    """
    from pca import fit_pca_probe, sweep_single_pc_interventions

    rows = []

    for k in subspace_dims:
        print(f"\n===== Comparing k={k} =====")

        # PCA bundle
        pca_bundle = fit_pca_probe(
            train_items=train_items,
            test_items=test_items,
            layer_idx=layer_idx,
            n_components=max(k, 1),
        )

        # supervised bundle
        sup_bundle = fit_supervised_subspace(
            train_items=train_items,
            test_items=test_items,
            layer_idx=layer_idx,
            subspace_dim=max(k, 1),
        )

        # sweep all coordinates up to k
        for coord_idx in range(k):
            pca_df = sweep_single_pc_interventions(
                bundle=pca_bundle,
                pc_idx=coord_idx,
                deltas=deltas,
                max_items=max_items,
            )

            sup_df = sweep_single_supervised_coord(
                bundle=sup_bundle,
                coord_idx=coord_idx,
                deltas=deltas,
                max_items=max_items,
            )

            pca_summary = (
                pca_df.groupby("delta")[["flipped", "correct_after", "prob_true_after"]]
                .mean()
                .reset_index()
            )
            pca_summary["method"] = "pca"
            pca_summary["coord_idx"] = coord_idx
            pca_summary["k"] = k

            sup_summary = (
                sup_df.groupby("delta")[["flipped", "correct_after", "prob_true_after"]]
                .mean()
                .reset_index()
            )
            sup_summary["method"] = "supervised"
            sup_summary["coord_idx"] = coord_idx
            sup_summary["k"] = k

            rows.append(pca_summary)
            rows.append(sup_summary)

    comparison_df = pd.concat(rows, ignore_index=True)

    print("\n===== Mean comparison by method and k =====")
    print(
        comparison_df.groupby(["method", "k"])[["flipped", "correct_after", "prob_true_after"]]
        .mean()
        .round(4)
    )

    return comparison_df

def run_supervized(train_items, test_items):
    sup_bundle = fit_supervised_subspace(
        train_items=train_items,
        test_items=test_items,
        layer_idx=6,
        subspace_dim=5,
    )

    run_single_supervised_intervention(
        bundle=sup_bundle,
        test_index=0,
        coord_updates={0: 2.0}
    )

    sup_coord1_df = sweep_single_supervised_coord(
        bundle=sup_bundle,
        coord_idx=0,
        deltas=(-3, -2, -1, 1, 2, 3),
        max_items=500
    )

    comparison_df = compare_pca_vs_supervised(
        train_items=train_items,
        test_items=test_items,
        layer_idx=6,
        subspace_dims=(1, 2, 3, 5),
        deltas=(-3, -2, -1, 1, 2, 3),
        max_items=500,
    )

    return sup_coord1_df, comparison_df