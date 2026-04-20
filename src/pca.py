from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from probing import build_layer_xy
import pandas as pd
import numpy as np

"""
PCA using F0 feature exclusively
Simple classification task (logistic regression)
Compare the results on the PCA subspace with the one on the raw subspace
"""

def pca_classification(x_train, y_train, x_test, y_test):
    pca = PCA(n_components=4)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Train PCA shape:", x_train_pca.shape)
    print("Test PCA shape:", x_test_pca.shape)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train_pca, y_train)

    y_pred = clf.predict(x_test_pca)

    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    clf_raw = LogisticRegression(max_iter=2000)
    clf_raw.fit(x_train, y_train)

    y_pred_raw = clf_raw.predict(x_test)

    print("Raw 10-d contour accuracy:", accuracy_score(y_test, y_pred_raw))
    print(classification_report(y_test, y_pred_raw))

def pca_classification_aggregate(x_train, y_train, x_test, y_test):

    results = []

    for k in range(1, 11):
        pca = PCA(n_components=k)
        Xtr = pca.fit_transform(x_train)
        Xte = pca.transform(x_test)

        clf = LogisticRegression(max_iter=3000)
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)

        acc = accuracy_score(y_test, pred)
        var = pca.explained_variance_ratio_.sum()

        results.append((k, var, acc))
        print(f"k={k:2d} | var={var:.4f} | acc={acc:.4f}")

    print(pca.components_[0])
    print(pca.components_[1])



"""
PCA probing using hidden states from the model
"""


def run_hidden_pca(train_items, test_items, layer_idx=12, k_list=None):
    """
    Run PCA-subspace classification for one wav2vec layer.
    """
    if k_list is None:
        k_list = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 768]

    x_train, y_train = build_layer_xy(train_items, layer_idx)
    x_test, y_test = build_layer_xy(test_items, layer_idx)

    print(f"Layer {layer_idx}")
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)

    results = []

    for k in k_list:
        # raw baseline
        if k == x_train.shape[1]:
            clf = LogisticRegression(max_iter=3000)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            var = 1.0

            results.append({
                "layer": layer_idx,
                "k": k,
                "explained_variance": var,
                "accuracy": acc,
            })

            print(f"k={k:3d} | var={var:.4f} | acc={acc:.4f} | RAW")
            continue

        pca = PCA(n_components=k)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)

        clf = LogisticRegression(max_iter=3000)
        clf.fit(x_train_pca, y_train)
        y_pred = clf.predict(x_test_pca)

        acc = accuracy_score(y_test, y_pred)
        var = pca.explained_variance_ratio_.sum()

        results.append({
            "layer": layer_idx,
            "k": k,
            "explained_variance": var,
            "accuracy": acc,
        })

        print(f"k={k:3d} | var={var:.4f} | acc={acc:.4f}")

    return results

def run_multi_layer_pca(train_items, test_items, layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), k_list=None):
    """
    Run PCA sweep for multiple layers.
    Returns a tidy DataFrame.
    """
    if k_list is None:
        k_list = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 768]

    all_rows = []

    for layer_idx in layers:
        print(f"\n===== Layer {layer_idx} =====")
        results = run_hidden_pca(
            train_items=train_items,
            test_items=test_items,
            layer_idx=layer_idx,
            k_list=k_list,
        )

        for r in results:
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    return df




"""
PCA intervention
	1.	fitting PCA + probe on one hidden layer
	2.	intervening on PCA coordinates
	3.	checking whether the probe prediction flips
	4.	sweeping a PC across many items
"""


def fit_pca_probe(train_items, test_items, layer_idx=6, n_components=100):
    """
    Fit PCA on train hidden states for one layer,
    then fit a logistic-regression probe on the original hidden states.

    Returns a bundle used for intervention.
    """
    x_train, y_train = build_layer_xy(train_items, layer_idx)
    x_test, y_test = build_layer_xy(test_items, layer_idx)

    pca = PCA(n_components=n_components)
    pca.fit(x_train)

    probe = LogisticRegression(max_iter=3000)
    probe.fit(x_train, y_train)

    y_pred = probe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Layer {layer_idx} | PCA dims = {n_components}")
    print("Probe accuracy on original hidden states:", acc)
    print(classification_report(y_test, y_pred))

    return {
        "layer_idx": layer_idx,
        "n_components": n_components,
        "pca": pca,
        "probe": probe,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_items": train_items,
        "test_items": test_items,
    }


def predict_probe(probe, x):
    """
    Predict tone and probabilities from one hidden vector.
    """
    pred = int(probe.predict(x.reshape(1, -1))[0])
    probs = probe.predict_proba(x.reshape(1, -1))[0]
    return pred, probs


def intervene_pca_coordinates(x, pca, coord_updates):
    """
    Intervene on selected PCA coordinates.

    Parameters
    ----------
    x : np.ndarray of shape [D]
        Original hidden vector.
    pca : fitted PCA
    coord_updates : dict
        Example: {0: +2.0, 1: -1.0}
        Means add +2.0 to PC1 score and -1.0 to PC2 score.

    Returns
    -------
    z_orig : original PCA coordinates
    z_mod  : modified PCA coordinates
    x_rec  : reconstructed hidden vector after intervention
    """
    z_orig = pca.transform(x.reshape(1, -1))[0]
    z_mod = z_orig.copy()

    for pc_idx, delta in coord_updates.items():
        z_mod[pc_idx] += delta

    x_rec = pca.inverse_transform(z_mod.reshape(1, -1))[0]
    return z_orig, z_mod, x_rec


def run_single_pca_intervention(bundle, test_index=0, coord_updates=None):
    """
    Apply one PCA intervention to one test item and compare
    probe predictions before vs after.

    Example:
        run_single_pca_intervention(bundle, test_index=5, coord_updates={0: 2.0})
    """
    if coord_updates is None:
        coord_updates = {0: 2.0}

    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    probe = bundle["probe"]
    pca = bundle["pca"]
    test_items = bundle["test_items"]

    x = x_test[test_index]
    true_tone = int(y_test[test_index])
    item = test_items[test_index]

    pred_before, probs_before = predict_probe(probe, x)
    z_orig, z_mod, x_mod = intervene_pca_coordinates(x, pca, coord_updates)
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

# Why deltas?
# (Why use values like -3, -2, -1, +1, +2, +3?)

# Because PCA coordinates are usually roughly standardized-ish in spread, so these are like:
#	•	small move
#	•	medium move
#	•	large move

# => testing sensitivity.


def sweep_single_pc_interventions(bundle, pc_idx=0, deltas=(-3, -2, -1, 1, 2, 3), max_items=500):
    """
    Sweep interventions on a single PC across many test items.

    Returns a DataFrame with one row per (item, delta).
    """
    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    probe = bundle["probe"]
    pca = bundle["pca"]
    test_items = bundle["test_items"]

    n = min(max_items, len(x_test))
    rows = []

    for i in range(n):
        x = x_test[i]
        true_tone = int(y_test[i])
        item = test_items[i]

        pred_before, probs_before = predict_probe(probe, x)

        for delta in deltas:
            z_orig, z_mod, x_mod = intervene_pca_coordinates(x, pca, {pc_idx: delta})
            pred_after, probs_after = predict_probe(probe, x_mod)

            rows.append({
                "test_index": i,
                "wav_file": item["wav_file"].name,
                "speaker": item["speaker"],
                "sound": item["sound"],
                "true_tone": true_tone,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "pc_idx": pc_idx,
                "delta": delta,
                "flipped": int(pred_before != pred_after),
                "correct_before": int(pred_before == true_tone),
                "correct_after": int(pred_after == true_tone),
                "prob_true_before": float(probs_before[true_tone - 1]),
                "prob_true_after": float(probs_after[true_tone - 1]),
            })

    df = pd.DataFrame(rows)

    print(f"\nPC{pc_idx+1} intervention summary")
    print(df.groupby("delta")[["flipped", "correct_before", "correct_after"]].mean())

    return df


def sweep_multiple_pcs(bundle, pc_list=(0, 1, 2, 3, 4), deltas=(-3, -2, -1, 1, 2, 3), max_items=500):
    """
    Run single-PC intervention sweeps for multiple PCs.
    """
    dfs = []

    for pc_idx in pc_list:
        df = sweep_single_pc_interventions(
            bundle=bundle,
            pc_idx=pc_idx,
            deltas=deltas,
            max_items=max_items,
        )
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    print("\nOverall summary by PC and delta")
    print(out.groupby(["pc_idx", "delta"])[["flipped", "correct_before", "correct_after"]].mean())

    return out

def run_pca_intervention(train_items, test_items):
    bundle = fit_pca_probe(
        train_items=train_items,
        test_items=test_items,
        layer_idx=6,
        n_components=100,
    )

    single_result = run_single_pca_intervention(
        bundle=bundle,
        test_index=0,
        coord_updates={0: 2.0}
    )

    pc1_df = sweep_single_pc_interventions(
        bundle=bundle,
        pc_idx=0,
        deltas=(-3, -2, -1, 1, 2, 3),
        max_items=500
    )

    pc2_df = sweep_single_pc_interventions(
        bundle=bundle,
        pc_idx=1,
        deltas=(-3, -2, -1, 1, 2, 3),
        max_items=500
    )

    all_pc_df = sweep_multiple_pcs(
        bundle=bundle,
        pc_list=(0, 1, 2, 3, 4),
        deltas=(-3, -2, -1, 1, 2, 3),
        max_items=500
    )

    return all_pc_df


# Analysis


def transition_table(df, pc_idx=None, delta=None, normalize=True):
    """
    Build tone transition matrix:
    pred_before -> pred_after

    or restricted to one PC / delta.
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

    print(f"\nTransition table | PC={pc_idx} | delta={delta}")
    print(tab.round(3))
    return tab


def true_to_after_table(df, pc_idx=None, delta=None, normalize=True):
    """
    True tone -> pred_after
    Useful for seeing directional tone movement.
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

    print(f"\nTrue -> After | PC={pc_idx} | delta={delta}")
    print(tab.round(3))
    return tab


def rank_pcs_by_flip(all_pc_df):
    """
    Average flip rate by PC.
    """
    summary = (
        all_pc_df.groupby("pc_idx")[["flipped", "correct_after"]]
        .mean()
        .sort_values("flipped", ascending=False)
    )

    print("\nRank PCs by intervention strength")
    print(summary.round(4))
    return summary

def run_pca_intervention_analysis(all_pc_df):
    transition_table(all_pc_df, pc_idx=4, delta=3)
    transition_table(all_pc_df, pc_idx=0, delta=3)
    true_to_after_table(all_pc_df, pc_idx=4, delta=3)
    rank_pcs_by_flip(all_pc_df)


    