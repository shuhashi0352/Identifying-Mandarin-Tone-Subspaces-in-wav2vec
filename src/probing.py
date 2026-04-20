from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_wav(wav_path):
    """
    Load one already-standardized wav file.
    Assumes:
    - mono
    - 16kHz
    """
    waveform, sr = torchaudio.load(str(wav_path))

    if waveform.shape[0] != 1:
        raise ValueError(f"{wav_path} is not mono as expected.")
    if sr != 16000:
        raise ValueError(f"{wav_path} is not 16kHz as expected.")

    waveform = waveform.squeeze(0)  # [time]
    return waveform, sr


def pool_hidden_states_in_interval(hidden_states, duration_sec, start_time, end_time):
    """
    Mean-pool each layer using only frames inside [start_time, end_time].

    hidden_states:
        tuple of tensors, each [1, T, D]

    Returns:
        pooled_layers: list of np.ndarray, each [D]
    """
    pooled_layers = []
    for layer_h in tqdm(hidden_states, desc="pooling", unit="pooling"):
        # [1, T, D] -> [T, D]
        layer_h = layer_h.squeeze(0)
        T = layer_h.shape[0]

        # approximate frame timestamps
        frame_times = np.linspace(0.0, duration_sec, num=T)

        mask = (frame_times >= start_time) & (frame_times <= end_time)

        # fallback: if no frame lands inside interval, use nearest frames
        if mask.sum() == 0:
            center = (start_time + end_time) / 2.0
            nearest_idx = np.argmin(np.abs(frame_times - center))
            mask[nearest_idx] = True

        selected = layer_h[mask]   # [t_sel, D]
        pooled = selected.mean(dim=0)  # [D]

        pooled_layers.append(pooled.detach().cpu().numpy())

    return pooled_layers

def build_metadata_map(cfg):
    meta = pd.read_csv(cfg["data"]["metadata_path"]).copy()

    audio_col = cfg["data"]["audio_col"]
    speaker_col = cfg["data"]["speaker_col"]
    label_col = cfg["data"]["label_col"]
    sound_col = cfg["data"]["sound_col"]

    meta["wav_file"] = meta[audio_col].apply(lambda x: Path(x).name)

    meta_map = (
        meta[["wav_file", speaker_col, label_col, sound_col]]
        .drop_duplicates()
        .set_index("wav_file")
        .to_dict("index")
    )

    return meta_map

def extract_hidden_states(cfg, resampled, model_name="facebook/wav2vec2-base", device=None):
    """
    Extract wav2vec hidden states and mean-pool only over the tone-bearing interval.

    resampled:
        your processed usable items, each containing:
            wav_file
            start_time
            end_time
            speaker (optional)
            tone (optional)
            sound (optional)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_map = build_metadata_map(cfg)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    hs_results = []

    with torch.no_grad():
        for item in tqdm(resampled, desc="extraction", unit="extraction"):
            wav_path = Path(item["wav_file"])
            wav_name = wav_path.name

            if wav_name not in meta_map:
                print(f"Skipping {wav_name}: not found in metadata")
                continue

            start_time = float(item["start_time"])
            end_time = float(item["end_time"])

            waveform, sr = load_wav(wav_path)
            duration_sec = len(waveform) / sr

            inputs = processor(
                waveform.numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=False,
            )

            input_values = inputs["input_values"].to(device)

            outputs = model(
                input_values,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states

            pooled_layers = pool_hidden_states_in_interval(
                hidden_states=hidden_states,
                duration_sec=duration_sec,
                start_time=start_time,
                end_time=end_time,
            )

            hs_results.append({
                "wav_file": wav_path,
                "speaker": meta_map[wav_name][cfg["data"]["speaker_col"]],
                "tone": meta_map[wav_name][cfg["data"]["label_col"]],
                "sound": meta_map[wav_name][cfg["data"]["sound_col"]],
                "start_time": start_time,
                "end_time": end_time,
                "layer_vectors": pooled_layers,
                "n_layers": len(pooled_layers),
                "hidden_dim": pooled_layers[0].shape[0],
            })

    print("num files:", len(hs_results))
    print("num layers:", hs_results[0]["n_layers"])
    print("hidden dim:", hs_results[0]["hidden_dim"])
    print("layer 0 shape:", hs_results[0]["layer_vectors"][0].shape)

    return hs_results

def split_hidden_states_by_metadata(hs_results, train_df, test_df):
    train_files = set(train_df["wav_file"])
    test_files = set(test_df["wav_file"])

    train_items = [x for x in hs_results if x["wav_file"].name in train_files]
    test_items = [x for x in hs_results if x["wav_file"].name in test_files]

    return train_items, test_items

def build_layer_xy(items, layer_idx):
    x = np.stack([x["layer_vectors"][layer_idx] for x in items])
    y = np.array([x["tone"] for x in items], dtype=int)
    return x, y

def run_split_hs(hs_results, train_df, test_df):
    train_items, test_items = split_hidden_states_by_metadata(hs_results, train_df, test_df)
    x_train_l0, y_train = build_layer_xy(train_items, layer_idx=0)
    x_test_l0, y_test = build_layer_xy(test_items, layer_idx=0)

    print(x_train_l0.shape, y_train.shape)
    print(x_test_l0.shape, y_test.shape)

    return x_train_l0, y_train, x_test_l0, y_test, train_items, test_items

def run_layerwise_probe(train_items, test_items):
    n_layers = train_items[0]["n_layers"]
    results = []

    for layer_idx in range(n_layers):
        x_train, y_train = build_layer_xy(train_items, layer_idx)
        x_test, y_test = build_layer_xy(test_items, layer_idx)

        clf = LogisticRegression(max_iter=3000)
        print(y_train[:10])
        print(type(y_train[0]))
        print(set(y_train[:20]))
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        results.append({
            "layer": layer_idx,
            "accuracy": acc,
        })

        print(f"Layer {layer_idx:2d} | acc = {acc:.4f}")

    return results