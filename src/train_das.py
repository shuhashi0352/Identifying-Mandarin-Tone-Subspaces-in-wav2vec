from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_label(item: Dict[str, Any]) -> int:
    for key in ["label", "tone", "tone_label", "y"]:
        if key in item:
            return int(item[key])
    raise KeyError(f"Could not find label key in item. Keys: {list(item.keys())}")


def _get_hidden_by_layer(item: Dict[str, Any], layer_idx: int) -> torch.Tensor:
    """
    Tries to extract one layer vector from your existing train_items/test_items.

    Expected common structures:
      item["hidden_states"][layer_idx] -> [D]
      item["pooled_hidden_states"][layer_idx] -> [D]
      item["layers"][layer_idx] -> [D]
    """
    for key in ["layer_vectors", "hidden_states", "pooled_hidden_states", "layer_hidden_states", "layers", "hs"]:
        if key in item:
            hs = item[key]
            h = hs[layer_idx]
            return torch.as_tensor(h).float().view(-1)

    raise KeyError(
        f"Could not find hidden-state key in item. Keys: {list(item.keys())}"
    )


def items_to_X_y(
    items: List[Dict[str, Any]],
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = []
    y = []

    for item in items:
        X.append(_get_hidden_by_layer(item, layer_idx))
        y.append(_get_label(item))

    X = torch.stack(X, dim=0)
    y = torch.tensor(y, dtype=torch.long)

    # Normalize labels to 0..C-1.
    unique = sorted(y.unique().tolist())
    label_map = {old: new for new, old in enumerate(unique)}
    y = torch.tensor([label_map[int(v)] for v in y], dtype=torch.long)

    print(f"[DAS] Layer {layer_idx} X shape: {tuple(X.shape)}")
    print(f"[DAS] Original labels: {unique}")
    print(f"[DAS] Internal label map: {label_map}")

    return X, y


class ToneDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SameSoundSpeakerPairDataset(Dataset):
    """
    Same-sound, same-speaker, different-tone DAS pairs.

    Correct examples:
        ma1_FV1 -> ma2_FV1
        ma1_FV1 -> ma3_FV1
        ma1_FV1 -> ma4_FV1

    Invalid examples excluded:
        ma1_FV1 -> ma1_FV1   # same tone
        ma1_FV1 -> ma4_FV2   # different speaker
        ma1_FV1 -> ma1_FV2   # different speaker and same tone

    Each item returns:
        h_base, y_base, h_source, y_source

    DAS objective:
        after replacing k DAS dimensions from source into base,
        classifier should predict y_source.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        items: List[Dict[str, Any]],
        seed: int = 0,
        sound_key: str = "sound",
        speaker_key: str = "speaker",
        use_all_pairs: bool = True,
        max_pairs_per_transition: int | None = None,
    ):
        self.X = X.float()
        self.y = y.long()
        self.items = items
        self.rng = np.random.default_rng(seed)

        self.sound_key = sound_key
        self.speaker_key = speaker_key
        self.use_all_pairs = use_all_pairs
        self.max_pairs_per_transition = max_pairs_per_transition

        if len(self.X) != len(self.items):
            raise ValueError(
                f"X/items length mismatch: len(X)={len(self.X)}, len(items)={len(self.items)}"
            )

        self.group_tone_to_indices = self._build_index()
        self.pairs = self._make_pairs()

        if len(self.pairs) == 0:
            raise ValueError(
                "No same-sound same-speaker different-tone pairs were created. "
                "Check whether each (sound, speaker) group has multiple tones."
            )

        print(f"[DAS pairs] Created {len(self.pairs)} same-sound-same-speaker pairs")

    def _build_index(self) -> Dict[Tuple[str, str], Dict[int, List[int]]]:
        """
        Build:

            (sound, speaker) -> tone -> [indices]

        Example:
            ('ma', 'FV1') -> {
                0: [idx for ma1_FV1],
                1: [idx for ma2_FV1],
                2: [idx for ma3_FV1],
                3: [idx for ma4_FV1],
            }
        """
        index: Dict[Tuple[str, str], Dict[int, List[int]]] = {}

        for i, item in enumerate(self.items):
            if self.sound_key not in item:
                raise KeyError(
                    f"Could not find sound key '{self.sound_key}' in item. "
                    f"Available keys: {list(item.keys())}"
                )

            if self.speaker_key not in item:
                raise KeyError(
                    f"Could not find speaker key '{self.speaker_key}' in item. "
                    f"Available keys: {list(item.keys())}"
                )

            sound = str(item[self.sound_key])
            speaker = str(item[self.speaker_key])
            tone = int(self.y[i].item())

            group = (sound, speaker)

            if group not in index:
                index[group] = {}

            if tone not in index[group]:
                index[group][tone] = []

            index[group][tone].append(i)

        return index

    def _make_pairs(self):
        pairs = []
        transition_counts = {}
        group_counts = {
            "groups_total": 0,
            "groups_with_at_least_2_tones": 0,
            "groups_with_all_4_tones": 0,
        }

        for (sound, speaker), tone_to_indices in self.group_tone_to_indices.items():
            group_counts["groups_total"] += 1

            tones = sorted(tone_to_indices.keys())

            # Need at least two tones to make different-tone pairs.
            if len(tones) < 2:
                continue

            group_counts["groups_with_at_least_2_tones"] += 1

            if len(tones) == 4:
                group_counts["groups_with_all_4_tones"] += 1

            for y_base in tones:
                for y_source in tones:
                    if y_base == y_source:
                        continue

                    base_indices = list(tone_to_indices[y_base])
                    source_indices = list(tone_to_indices[y_source])

                    # All unique directed base/source index combinations.
                    candidate_pairs = [
                        (int(i_base), int(i_source))
                        for i_base in base_indices
                        for i_source in source_indices
                        if int(i_base) != int(i_source)
                    ]

                    if len(candidate_pairs) == 0:
                        continue

                    if self.use_all_pairs:
                        chosen_pairs = candidate_pairs
                    else:
                        if self.max_pairs_per_transition is None:
                            raise ValueError(
                                "max_pairs_per_transition must be set when use_all_pairs=False"
                            )

                        n_sample = min(
                            self.max_pairs_per_transition,
                            len(candidate_pairs),
                        )

                        chosen_idx = self.rng.choice(
                            len(candidate_pairs),
                            size=n_sample,
                            replace=False,
                        )

                        chosen_pairs = [candidate_pairs[int(j)] for j in chosen_idx]

                    pairs.extend(chosen_pairs)

                    transition_key = f"{y_base}->{y_source}"
                    transition_counts[transition_key] = (
                        transition_counts.get(transition_key, 0) + len(chosen_pairs)
                    )

        self.rng.shuffle(pairs)

        print("[DAS pairs] Group counts:")
        for key, value in group_counts.items():
            print(f"  {key}: {value}")

        print("[DAS pairs] Same-sound same-speaker pairs by transition:")
        for key, count in sorted(transition_counts.items()):
            print(f"  {key}: {count} pairs")

        print("[DAS pairs] Example pairs:")
        for i_base, i_source in pairs[:10]:
            base_item = self.items[i_base]
            source_item = self.items[i_source]

            print(
                f"  "
                f"{base_item[self.sound_key]} | "
                f"speaker={base_item[self.speaker_key]} | "
                f"{int(self.y[i_base].item())}->{int(self.y[i_source].item())} | "
                f"indices {i_base}->{i_source}"
            )

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i_base, i_source = self.pairs[idx]

        h_base = self.X[i_base]
        y_base = self.y[i_base]

        h_source = self.X[i_source]
        y_source = self.y[i_source]

        return h_base, y_base, h_source, y_source


class LinearToneClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


class OrthogonalDASRotation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        A = torch.eye(dim) + 0.01 * torch.randn(dim, dim)
        self.A = nn.Parameter(A)

    def rotation(self) -> torch.Tensor:
        Q, R = torch.linalg.qr(self.A)

        diag = torch.sign(torch.diag(R))
        diag[diag == 0] = 1
        Q = Q * diag.unsqueeze(0)

        return Q

    def intervene(
        self,
        h_base: torch.Tensor,
        h_source: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        R = self.rotation()

        z_base = h_base @ R
        z_source = h_source @ R

        z_mix = z_base.clone()
        z_mix[:, :k] = z_source[:, :k]

        h_mix = z_mix @ R.T
        return h_mix


@torch.no_grad()
def eval_classifier_acc(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    logits = model(X.to(device))
    pred = logits.argmax(dim=-1).cpu()
    return accuracy_score(y.cpu().numpy(), pred.numpy())


def train_classifier(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> LinearToneClassifier:
    dim = X_train.shape[1]
    model = LinearToneClassifier(dim=dim, num_classes=num_classes).to(device)

    loader = DataLoader(
        ToneDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for h, y in loader:
            h = h.to(device)
            y = y.to(device)

            logits = model(h)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(h)

        test_acc = eval_classifier_acc(model, X_test, y_test, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[DAS classifier] epoch={epoch:03d} "
                f"loss={total_loss / len(X_train):.4f} "
                f"test_acc={test_acc:.4f}"
            )

    model.load_state_dict(best_state)
    print(f"[DAS classifier] best test acc={best_acc:.4f}")

    return model


@torch.no_grad()
def evaluate_das(
    classifier: LinearToneClassifier,
    das: OrthogonalDASRotation,
    pair_dataset: Dataset,
    k: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    classifier.eval()
    das.eval()

    loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=False)

    y_base_all = []
    y_source_all = []
    pred_base_all = []
    pred_mix_all = []

    transition_records = []

    for h_base, y_base, h_source, y_source in loader:
        h_base = h_base.to(device)
        h_source = h_source.to(device)

        logits_base = classifier(h_base)
        pred_base = logits_base.argmax(dim=-1).cpu()

        h_mix = das.intervene(h_base, h_source, k=k)
        logits_mix = classifier(h_mix)
        pred_mix = logits_mix.argmax(dim=-1).cpu()

        y_base_cpu = y_base.cpu()
        y_source_cpu = y_source.cpu()

        y_base_all.append(y_base_cpu)
        y_source_all.append(y_source_cpu)
        pred_base_all.append(pred_base)
        pred_mix_all.append(pred_mix)

        for b, s, pb, pm in zip(
            y_base_cpu.tolist(),
            y_source_cpu.tolist(),
            pred_base.tolist(),
            pred_mix.tolist(),
        ):
            transition_records.append((b, s, pb, pm))

    y_base = torch.cat(y_base_all).numpy()
    y_source = torch.cat(y_source_all).numpy()
    pred_base = torch.cat(pred_base_all).numpy()
    pred_mix = torch.cat(pred_mix_all).numpy()

    flip_rate = float(np.mean(pred_mix != pred_base))
    target_success_rate = float(np.mean(pred_mix == y_source))

    base_correct = pred_base == y_base
    if base_correct.sum() > 0:
        target_success_given_base_correct = float(
            np.mean(pred_mix[base_correct] == y_source[base_correct])
        )
    else:
        target_success_given_base_correct = float("nan")

    transition_metrics = {}
    for b in sorted(set(y_base.tolist())):
        for s in sorted(set(y_source.tolist())):
            if b == s:
                continue

            mask = (y_base == b) & (y_source == s)
            if mask.sum() == 0:
                continue

            transition_metrics[f"{b}->{s}"] = {
                "n": int(mask.sum()),
                "flip_rate": float(np.mean(pred_mix[mask] != pred_base[mask])),
                "target_success_rate": float(np.mean(pred_mix[mask] == y_source[mask])),
            }

    return {
        "k": int(k),
        "num_pairs": int(len(pair_dataset)),
        "flip_rate": flip_rate,
        "target_success_rate": target_success_rate,
        "target_success_given_base_correct": target_success_given_base_correct,
        "transition_metrics": transition_metrics,
        "confusion_matrix_rows_source_gold_cols_intervened_pred": confusion_matrix(
            y_source,
            pred_mix,
        ).tolist(),
        "classification_report_source_gold_vs_intervened_pred": classification_report(
            y_source,
            pred_mix,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_das_rotation(
    classifier: LinearToneClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    train_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    k: int,
    batch_size: int,
    epochs: int,
    lr: float,
    pairs_per_direction_train: int,
    pairs_per_direction_test: int,
    seed: int,
    device: torch.device,
) -> Tuple[OrthogonalDASRotation, Dict[str, Any]]:
    dim = X_train.shape[1]
    das = OrthogonalDASRotation(dim=dim).to(device)

    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    train_pairs = SameSoundSpeakerPairDataset(
        X=X_train,
        y=y_train,
        items=train_items,
        seed=seed,
        use_all_pairs=True,
    )

    test_pairs = SameSoundSpeakerPairDataset(
        X=X_test,
        y=y_test,
        items=test_items,
        seed=seed + 1,
        use_all_pairs=True,
    )

    loader = DataLoader(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(das.parameters(), lr=lr, weight_decay=1e-5)

    best_success = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        das.train()
        total_loss = 0.0

        for h_base, y_base, h_source, y_source in loader:
            h_base = h_base.to(device)
            h_source = h_source.to(device)
            y_source = y_source.to(device)

            h_mix = das.intervene(h_base, h_source, k=k)
            logits = classifier(h_mix)

            loss = F.cross_entropy(logits, y_source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(h_base)

        metrics = evaluate_das(
            classifier=classifier,
            das=das,
            pair_dataset=test_pairs,
            k=k,
            batch_size=batch_size,
            device=device,
        )

        success = metrics["target_success_rate"]

        if success > best_success:
            best_success = success
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in das.state_dict().items()
            }
            best_metrics = metrics

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[DAS] epoch={epoch:03d} "
                f"loss={total_loss / len(train_pairs):.4f} "
                f"target_success={metrics['target_success_rate']:.4f} "
                f"flip={metrics['flip_rate']:.4f}"
            )

    das.load_state_dict(best_state)
    print(f"[DAS] best target_success={best_success:.4f}")

    return das, best_metrics


def run_das_best_layer(
    train_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    layer_idx: int = 6,
    k: int = 16,
) -> Dict[str, Any]:
    """
    Main callable function for src/main.py.
    """

    das_cfg = cfg.get("das", {})

    seed = int(das_cfg.get("seed", 42))
    batch_size = int(das_cfg.get("batch_size", 64))
    epochs_clf = int(das_cfg.get("epochs_clf", 100))
    epochs_das = int(das_cfg.get("epochs_das", 200))
    lr_clf = float(das_cfg.get("lr_clf", 1e-3))
    lr_das = float(das_cfg.get("lr_das", 5e-4))
    pairs_per_direction_train = int(das_cfg.get("pairs_per_direction_train", 500))
    pairs_per_direction_test = int(das_cfg.get("pairs_per_direction_test", 100))
    out_dir = Path(das_cfg.get("out_dir", f"results/das_layer{layer_idx}_k{k}"))

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAS] device={device}")

    X_train, y_train = items_to_X_y(train_items, layer_idx=layer_idx)
    X_test, y_test = items_to_X_y(test_items, layer_idx=layer_idx)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_classes = int(max(y_train.max().item(), y_test.max().item()) + 1)

    classifier = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        batch_size=batch_size,
        epochs=epochs_clf,
        lr=lr_clf,
        device=device,
    )

    das, metrics = train_das_rotation(
        classifier=classifier,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_items=train_items,
        test_items=test_items,
        k=k,
        batch_size=batch_size,
        epochs=epochs_das,
        lr=lr_das,
        pairs_per_direction_train=pairs_per_direction_train,
        pairs_per_direction_test=pairs_per_direction_test,
        seed=seed,
        device=device,
    )

    save_obj = {
        "layer_idx": layer_idx,
        "k": k,
        "metrics": metrics,
        "mean": mean,
        "std": std,
        "classifier_state_dict": classifier.state_dict(),
        "das_state_dict": das.state_dict(),
        "das_cfg": das_cfg,
    }

    torch.save(save_obj, out_dir / "checkpoint.pt")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[DAS] saved checkpoint: {out_dir / 'checkpoint.pt'}")
    print(f"[DAS] saved metrics:    {out_dir / 'metrics.json'}")

    return metrics

def run_das_layer_sweep(
    train_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run same-sound DAS over multiple layers and k values.

    Expected config:

    das:
      seed: 42
      batch_size: 64
      epochs_clf: 50
      epochs_das: 30
      lr_clf: 0.001
      lr_das: 0.0005
      pairs_per_direction_train: 5
      pairs_per_direction_test: 5
      layers: [0, 2, 4, 6, 8, 10, 12]
      k_values: [4]
      out_dir: ./results/das_layer_sweep
    """

    das_cfg = cfg.get("das", {})

    seed = int(das_cfg.get("seed", 42))
    batch_size = int(das_cfg.get("batch_size", 64))
    epochs_clf = int(das_cfg.get("epochs_clf", 50))
    epochs_das = int(das_cfg.get("epochs_das", 30))
    lr_clf = float(das_cfg.get("lr_clf", 1e-3))
    lr_das = float(das_cfg.get("lr_das", 5e-4))
    pairs_per_direction_train = int(das_cfg.get("pairs_per_direction_train", 5))
    pairs_per_direction_test = int(das_cfg.get("pairs_per_direction_test", 5))

    layers = das_cfg.get("layers", [0, 2, 4, 6, 8, 10, 12])
    k_values = das_cfg.get("k_values", [4])

    layers = [int(x) for x in layers]
    k_values = [int(x) for x in k_values]

    out_dir = Path(das_cfg.get("out_dir_sweep", "./results/das_layer_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAS sweep] device={device}")
    print(f"[DAS sweep] layers={layers}")
    print(f"[DAS sweep] k_values={k_values}")

    all_results: List[Dict[str, Any]] = []

    for layer_idx in layers:
        print("\n" + "=" * 80)
        print(f"[DAS sweep] Starting layer {layer_idx}")
        print("=" * 80)

        # -------------------------
        # 1. Extract layer vectors
        # -------------------------
        X_train, y_train = items_to_X_y(train_items, layer_idx=layer_idx)
        X_test, y_test = items_to_X_y(test_items, layer_idx=layer_idx)

        # -------------------------
        # 2. Standardize by train stats
        # -------------------------
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)

        X_train_std = (X_train - mean) / std
        X_test_std = (X_test - mean) / std

        num_classes = int(max(y_train.max().item(), y_test.max().item()) + 1)

        # -------------------------
        # 3. Train classifier once per layer
        # -------------------------
        classifier = train_classifier(
            X_train=X_train_std,
            y_train=y_train,
            X_test=X_test_std,
            y_test=y_test,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs_clf,
            lr=lr_clf,
            device=device,
        )

        classifier_test_acc = eval_classifier_acc(
            classifier,
            X_test_std,
            y_test,
            device,
        )

        print(
            f"[DAS sweep] layer={layer_idx} "
            f"classifier_test_acc={classifier_test_acc:.4f}"
        )

        # Save classifier for this layer
        layer_dir = out_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "layer_idx": layer_idx,
                "mean": mean,
                "std": std,
                "classifier_state_dict": classifier.state_dict(),
                "classifier_test_acc": classifier_test_acc,
                "das_cfg": das_cfg,
            },
            layer_dir / "classifier.pt",
        )

        # -------------------------
        # 4. Train DAS for each k
        # -------------------------
        for k in k_values:
            print("\n" + "-" * 80)
            print(f"[DAS sweep] Layer {layer_idx}, k={k}")
            print("-" * 80)

            das, metrics = train_das_rotation(
                classifier=classifier,
                X_train=X_train_std,
                y_train=y_train,
                X_test=X_test_std,
                y_test=y_test,
                train_items=train_items,
                test_items=test_items,
                k=k,
                batch_size=batch_size,
                epochs=epochs_das,
                lr=lr_das,
                pairs_per_direction_train=pairs_per_direction_train,
                pairs_per_direction_test=pairs_per_direction_test,
                seed=seed + layer_idx * 1000 + k,
                device=device,
            )

            metrics["layer_idx"] = int(layer_idx)
            metrics["k"] = int(k)
            metrics["classifier_test_acc"] = float(classifier_test_acc)

            all_results.append(metrics)

            k_dir = layer_dir / f"k_{k}"
            k_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "layer_idx": layer_idx,
                    "k": k,
                    "metrics": metrics,
                    "mean": mean,
                    "std": std,
                    "classifier_state_dict": classifier.state_dict(),
                    "das_state_dict": das.state_dict(),
                    "classifier_test_acc": classifier_test_acc,
                    "das_cfg": das_cfg,
                },
                k_dir / "checkpoint.pt",
            )

            with open(k_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # Save combined results after every run, so progress is not lost.
            with open(out_dir / "all_metrics.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            print(
                f"[DAS sweep] Saved layer={layer_idx}, k={k} metrics to "
                f"{k_dir / 'metrics.json'}"
            )

    # -------------------------
    # 5. Save compact CSV summary
    # -------------------------
    csv_path = out_dir / "summary.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "layer_idx,k,classifier_test_acc,target_success_rate,"
            "target_success_given_base_correct,flip_rate,num_pairs\n"
        )

        for m in all_results:
            f.write(
                f"{m['layer_idx']},"
                f"{m['k']},"
                f"{m['classifier_test_acc']},"
                f"{m['target_success_rate']},"
                f"{m['target_success_given_base_correct']},"
                f"{m['flip_rate']},"
                f"{m['num_pairs']}\n"
            )

    print(f"[DAS sweep] saved all metrics: {out_dir / 'all_metrics.json'}")
    print(f"[DAS sweep] saved summary CSV: {csv_path}")

    return all_results