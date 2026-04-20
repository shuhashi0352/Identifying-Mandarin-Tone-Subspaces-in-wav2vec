from pathlib import Path
import numpy as np
import parselmouth
import pandas as pd
import numpy as np
import yaml
from praatio import textgrid

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_f0_contour(wav_path, time_step, pitch_floor, pitch_ceiling):
    """
    Extract frame-level F0 contour from a wav file using Praat/parselmouth.

    Returns:
        times: np.ndarray of shape [T]
        f0:    np.ndarray of shape [T]
               unvoiced frames are np.nan
    """
    contours = []

    wav_path = Path(wav_path)
    wav_files = sorted(wav_path.glob("*.wav"))

    for wav_file in wav_files:
        snd = parselmouth.Sound(str(wav_file))
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )

        f0 = pitch.selected_array["frequency"].copy()
        times = pitch.xs()

        # Praat uses 0 for unvoiced frames -> convert to nan
        f0[f0 == 0] = np.nan

        contours.append((wav_file, times, f0))

    return contours

def parse_phone_intervals(textgrid_path):
    """
    Return non-empty phone intervals from one TextGrid file.
    """
    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    phone_tier = tg._tierDict["phones"]

    intervals = []
    for start, end, label in phone_tier.entries:
        label = label.strip()
        if label == "":
            continue
        intervals.append({
            "start": float(start),
            "end": float(end),
            "label": label
        })

    return intervals

def get_tone_bearing_interval(phone_intervals, vowel_set):
    """
    Keep from the first vowel to the last non-empty phone interval.
    """
    first_vowel_idx = None

    for i, interval in enumerate(phone_intervals):
        if interval["label"] in vowel_set:
            first_vowel_idx = i
            break

    if first_vowel_idx is None:
        raise ValueError("No vowel found in phone intervals.")

    start_time = phone_intervals[first_vowel_idx]["start"]
    end_time = phone_intervals[-1]["end"]

    return start_time, end_time

def filter_f0_by_interval(times, f0, start_time, end_time, pad=0.015):
    """
    Keep only F0 frames inside the tone-bearing interval.
    """
    mask = (times >= start_time - pad) & (times <= end_time + pad)
    return times[mask], f0[mask]


def summarize_skip_bias(cfg, skip_log, total_files):
    meta = pd.read_csv(cfg["data"]["metadata_path"]).copy()

    audio_col = cfg["data"]["audio_col"]
    label_col = cfg["data"]["label_col"]
    speaker_col = cfg["data"]["speaker_col"]
    sound_col = cfg["data"]["sound_col"]

    meta["wav_file"] = meta[audio_col].apply(lambda x: Path(x).name)

    skip_df = pd.DataFrame(skip_log)
    if skip_df.empty:
        print("No skipped files.")
        return None

    df = meta.merge(skip_df, on="wav_file", how="inner")

    print("===== Skip counts by reason =====")
    print(df["reason"].value_counts())
    print()

    print("===== Skips by speaker =====")
    print(pd.crosstab(df[speaker_col], df["reason"]))
    print()

    print("===== Skips by tone =====")
    print(pd.crosstab(df[label_col], df["reason"]))
    print()

    print("===== Most skipped syllable types =====")
    print(df[sound_col].value_counts().head(40))
    print()

    print("===== Missing TextGrid syllables =====")
    print(df.loc[df["reason"] == "missing_textgrid", sound_col].value_counts().head(40))
    print()

    print("===== No-voiced-F0 syllables =====")
    print(df.loc[df["reason"] == "no_voiced_f0", sound_col].value_counts().head(40))
    print()

    print(f"Total files in corpus: {total_files}")
    print(f"Total skipped files: {len(df)}")
    print(f"Skip rate: {len(df) / total_files:.3%}")

    return df


def extract_f0(cfg, time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0):
    """
    Extract F0 contours, then keep only the tone-bearing region
    determined from the matching TextGrid.
    """
    root = Path(cfg["data"]["root_dir"])
    wav_dir = root / "tone_perfect_wav"
    textgrid_dir = root / "mfa_textgrids"
    vowel_set = {"a", "e", "i", "o", "u", "y", "ə", "ɤ", "ɛ", "ɔ", "ɚ", "ʊ"}

    results = []
    skip_log = []

    contours = extract_f0_contour(
        wav_path=wav_dir,
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )

    total_files = len(contours)

    for wav_file, times, f0 in contours:
        tg_path = textgrid_dir / f"{wav_file.stem}.TextGrid"

        if not tg_path.exists():
            skip_log.append({
                "wav_file": wav_file.name,
                "reason": "missing_textgrid",
            })
            continue

        phone_intervals = parse_phone_intervals(tg_path)
        start_time, end_time = get_tone_bearing_interval(phone_intervals, vowel_set)
        kept_times, kept_f0 = filter_f0_by_interval(times, f0, start_time, end_time, pad=0.015)

        if len(kept_f0) == 0 or np.sum(~np.isnan(kept_f0)) == 0:
            skip_log.append({
                "wav_file": wav_file.name,
                "reason": "no_voiced_f0",
                "start_time": start_time,
                "end_time": end_time,
            })
            continue

        results.append({
            "wav_file": wav_file,
            "start_time": start_time,
            "end_time": end_time,
            "times": kept_times,
            "f0": kept_f0,
        })

    # skip_df = summarize_skip_bias(cfg, skip_log, total_files)

    return results # skip_df (If needed)