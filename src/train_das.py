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


class SameSoundPairDataset(Dataset):
    """
    Same-sound, different-tone DAS pairs.

    This creates controlled pairs like:

        ma1  -> ma2
        ma1  -> ma3
        ma1  -> ma4

    rather than uncontrolled pairs like:

        ma1  -> shi4

    Each item returns:
        h_base, y_base, h_source, y_source

    DAS objective:
        after replacing the first k DAS dimensions from source into base,
        the classifier should predict y_source.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        items: List[Dict[str, Any]],
        pairs_per_direction: int,
        seed: int = 0,
        sound_key: str = "sound",
    ):
        self.X = X.float()
        self.y = y.long()
        self.items = items
        self.pairs_per_direction = pairs_per_direction
        self.rng = np.random.default_rng(seed)
        self.sound_key = sound_key

        if len(self.X) != len(self.items):
            raise ValueError(
                f"X/items length mismatch: len(X)={len(self.X)}, len(items)={len(self.items)}"
            )

        self.sound_tone_to_indices = self._build_index()
        self.pairs = self._make_pairs()

        if len(self.pairs) == 0:
            raise ValueError(
                "No same-sound different-tone pairs were created. "
                "Check whether each sound appears with multiple tones."
            )

        print(f"[DAS pairs] Created {len(self.pairs)} same-sound pairs")

    def _build_index(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Build:

            sound -> tone -> [indices]

        Example:
            'ma' -> {
                0: [idxs for ma1],
                1: [idxs for ma2],
                2: [idxs for ma3],
                3: [idxs for ma4],
            }
        """
        index: Dict[str, Dict[int, List[int]]] = {}

        for i, item in enumerate(self.items):
            if self.sound_key not in item:
                raise KeyError(
                    f"Could not find sound key '{self.sound_key}' in item. "
                    f"Available keys: {list(item.keys())}"
                )

            sound = str(item[self.sound_key])
            tone = int(self.y[i].item())

            if sound not in index:
                index[sound] = {}

            if tone not in index[sound]:
                index[sound][tone] = []

            index[sound][tone].append(i)

        return index

    def _make_pairs(self):
        pairs = []
        transition_counts = {}

        for sound, tone_to_indices in self.sound_tone_to_indices.items():
            tones = sorted(tone_to_indices.keys())

            # Need at least two tones for this sound.
            if len(tones) < 2:
                continue

            for y_base in tones:
                for y_source in tones:
                    if y_base == y_source:
                        continue

                    base_indices = tone_to_indices[y_base]
                    source_indices = tone_to_indices[y_source]

                    transition_key = f"{y_base}->{y_source}"
                    transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1

                    for _ in range(self.pairs_per_direction):
                        i_base = int(self.rng.choice(base_indices))
                        i_source = int(self.rng.choice(source_indices))
                        pairs.append((i_base, i_source))

        self.rng.shuffle(pairs)
        print("[DAS pairs] Example same-sound pairs:")
        for i_base, i_source in pairs[:10]:
            base_item = self.items[i_base]
            source_item = self.items[i_source]

            print(
                f"  sound={base_item[self.sound_key]} | "
                f"{int(self.y[i_base].item())}->{int(self.y[i_source].item())}"
            )

        print("[DAS pairs] Available same-sound transition types:")
        for key, count in sorted(transition_counts.items()):
            print(f"  {key}: {count} sounds")

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

    train_pairs = SameSoundPairDataset(X_train, y_train, train_items, pairs_per_direction_train, seed)

    test_pairs = SameSoundPairDataset(X_test, y_test, test_items, pairs_per_direction_test, seed + 1)

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