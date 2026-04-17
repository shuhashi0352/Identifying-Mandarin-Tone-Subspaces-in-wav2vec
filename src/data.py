import requests
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import soundfile as sf

import pandas as pd
import yaml
import subprocess
from sklearn.model_selection import train_test_split

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def convert_wav(cfg):
    mp3_dir = Path(cfg["data"]["audio_mp3_dir"])
    
    # sorted for reproducability (sequence may not be stable depending on OS)
    mp3_files = sorted(mp3_dir.glob("*.mp3"))

    root = Path(cfg["data"]["root_dir"])
    wav_dir = root / "tone_perfect_wav"
    os.makedirs(wav_dir, exist_ok=True)

    for file in mp3_files:
        file_name = file.stem.replace("_MP3", "_WAV")
        output_path = wav_dir / f"{file_name}.wav"

        if output_path.exists():
            print(f"{output_path} already exists. Skip converting...")
            continue

        subprocess.run([
            "ffmpeg",
            "-i", str(file),
            "-ac", "1",
            "-ar", "16000",
            output_path
        ], check=True)
    
    return wav_dir

def parse_xml(xml_path):
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {
        "xml_path": str(xml_path),
        "sound": root.findtext("sound"),
        "tone": root.findtext("tone"),
        "pinyin": root.findtext("pinyin"),
        "speaker": root.findtext("speaker"),
        "gender": root.findtext("gender"),
        "identifier": root.findtext("identifier")
    }

    return data

def process_wav(wav_path):
    wav_path = Path(wav_path)
    info = sf.info(wav_path)

    data = {
        "wav_path": str(wav_path),
        "sample_rate": info.samplerate,
        "duration_sec": info.duration,
        "channels": info.channels
    }

    return data

def build_metadata(cfg, wav_dir):
    # From xml
    xml_dir = Path(cfg["data"]["audio_xml_dir"])
    xml_files = sorted(xml_dir.glob("*CUSTOM.xml"))

    # From audio processing
    wav_files = sorted(wav_dir.glob("*.wav"))

    xml_dict = {}
    for file in xml_files:
        xml_data = parse_xml(file)
        identifier = file.stem.replace("_CUSTOM", "")
        xml_dict[identifier] = xml_data
    
    wav_dict = {}
    for file in wav_files:
        wav_data = process_wav(file)
        identifier = file.stem.replace("_WAV", "")
        wav_dict[identifier] = wav_data

    rows = []
    all_ids = sorted(set(xml_dict) & set(wav_dict))

    for identifier in all_ids:
        row = {
            **xml_dict[identifier],
            **wav_dict[identifier],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("data/metadata.csv", index=False, encoding="utf-8")

    return df

def build_mfa_lab(df):
    wav_dir = Path("data/tone_perfect_wav")

    for _, row in df.iterrows():
        wav_csv_path = Path(row["wav_path"])
        lab_file = wav_dir / wav_csv_path.name
        
        # create transcript
        transcript = f"{row['sound']}"
        
        lab_path = lab_file.with_suffix(".lab")
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    
def split_df(cfg, df):

    text = cfg["data"]["text_col"]
    label = cfg["data"]["label_col"]
    train_size = cfg["experiment"]["train_size"]
    ratio_dev_test = cfg["experiment"]["ratio_dev_test"]
    seed = cfg["experiment"]["seed"]

    # Raise Error if data lacks any required columns
    missing = [c for c in [text, label] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    train, temporary = train_test_split(df, train_size=train_size, random_state=seed, stratify=df[label])
    test, dev = train_test_split(temporary, train_size=ratio_dev_test, random_state=seed, stratify=temporary[label])

    return train, dev, test, text, label

# def split_donor_receiver_df(df, label_col, donor_label=0, receiver_label=3):

#     # donor/receiver extraction
#     donor_df = df[df[label_col] == donor_label].copy()
#     receiver_df = df[df[label_col] == receiver_label].copy()

#     # Sanity check
#     if len(donor_df) == 0:
#         raise ValueError(f"No donor instances found for label={donor_label}")
#     if len(receiver_df) == 0:
#         raise ValueError(f"No receiver instances found for label={receiver_label}")

#     return donor_df, receiver_df # Don't return the base df since it won't be used for causality tests