from pathlib import Path
import pandas as pd

def build_metadata(cfg, resampled):
    """
    Keep only usable files and attach metadata.
    """
    meta = pd.read_csv(cfg["data"]["metadata_path"]).copy()

    audio_col = cfg["data"]["audio_col"]
    speaker_col = cfg["data"]["speaker_col"]
    label_col = cfg["data"]["label_col"]
    sound_col = cfg["data"]["sound_col"]

    meta["wav_file"] = meta[audio_col].apply(lambda x: Path(x).name)

    usable_df = pd.DataFrame({
        "wav_file": [item["wav_file"].name for item in resampled]
    })

    df = meta.merge(usable_df, on="wav_file", how="inner")

    # keep only needed columns
    df = df[[audio_col, "wav_file", speaker_col, label_col, sound_col]].drop_duplicates()

    return df

def speaker_stratified_split(df, speaker_col="speaker"):
    """
    Fixed train/test split by speaker.
    Test: FV3, MV3
    Train: all remaining speakers
    """
    test_speakers = {"FV3", "MV3"}
    all_speakers = set(df[speaker_col].unique())
    train_speakers = all_speakers - test_speakers

    train_df = df[df[speaker_col].isin(train_speakers)].reset_index(drop=True)
    test_df = df[df[speaker_col].isin(test_speakers)].reset_index(drop=True)

    return train_df, test_df

def run_stratification(cfg, resampled):
    usable_meta = build_metadata(cfg, resampled)

    speaker_col = cfg["data"]["speaker_col"]

    train_df, test_df = speaker_stratified_split(usable_meta, speaker_col=speaker_col)

    # Sanity check
    print("Train speakers:", sorted(train_df[speaker_col].unique()))
    print("Test speakers:", sorted(test_df[speaker_col].unique()))
    print("Overlap:", set(train_df[speaker_col]).intersection(set(test_df[speaker_col])))

    # label_col = cfg["data"]["label_col"]
    # sound_col = cfg["data"]["sound_col"]

    # print("Train tone counts")
    # print(train_df[label_col].value_counts().sort_index())
    # print()

    # print("Test tone counts")
    # print(test_df[label_col].value_counts().sort_index())
    # print()

    # print("Train speaker counts")
    # print(train_df[speaker_col].value_counts().sort_index())
    # print()

    # print("Test speaker counts")
    # print(test_df[speaker_col].value_counts().sort_index())

    return train_df, test_df