from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch
from tqdm import tqdm
from pathlib import Path
from praatio import textgrid
import numpy as np

def split_resampled(resampled, train_df, test_df):
    train_files = set(train_df["wav_file"])
    test_files = set(test_df["wav_file"])

    train_items = [item for item in resampled if item["wav_file"].name in train_files]
    test_items = [item for item in resampled if item["wav_file"].name in test_files]

    return train_items, test_items

def build_xy(items, metadata_df, tone_col="tone", sound_col="sound", speaker_col="speaker"):
    tone_map = dict(zip(metadata_df["wav_file"], metadata_df[tone_col]))
    sound_map = dict(zip(metadata_df["wav_file"], metadata_df[sound_col]))
    speaker_map = dict(zip(metadata_df["wav_file"], metadata_df[speaker_col]))

    x = np.stack([item["f0_z_resampled"] for item in items])
    y = np.array([tone_map[item["wav_file"].name] for item in items])

    wav_files = np.array([item["wav_file"].name for item in items])
    sounds = np.array([sound_map[item["wav_file"].name] for item in items])
    speakers = np.array([speaker_map[item["wav_file"].name] for item in items])

    return x, y, wav_files, sounds, speakers

def create_train_test(resampled, train_df, test_df):
    train_items, test_items = split_resampled(resampled, train_df, test_df)

    x_train, y_train, train_wavs, train_sounds, train_speakers = build_xy(train_items, train_df)
    x_test, y_test, test_wavs, test_sounds, test_speakers = build_xy(test_items, test_df)

    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    return x_train, y_train, x_test, y_test












def preprocess(df):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    data = []

    for _, row in df.dropna().iterrows():
        audio_path = row["wav_path"]
        id = row["identifier"]
        tone = row["tone"]
        speaker = row["speaker"]
        syllable = row["sound"]

        waveform, sr = torchaudio.load(audio_path)
        # Make them consistent with mono audio -> 1dim [num_sumples]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0) 

        tensors = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt"
        )

        data.append({
            "id": id,
            "tone": tone,
            "speaker": speaker,
            "syllable": syllable,
            "tensors": tensors
        })
        
    return data

def parse_text_interval(textgrid_path):
    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    tier = tg.tierDict["phones"]

def hs_extraction(df):
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()

    data = preprocess(df)
    results = []

    with torch.no_grad():
        for item in tqdm(data, desc="Extracting hidden states"):
            tensors = item["tensors"]
            outputs = model(**tensors, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            results.append({
                "id": item["id"],
                "tone": item["tone"],
                "speaker": item["speaker"],
                "syllable": item["syllable"],
                "hidden_states": hidden_states,
            })

        print(results)

    return hidden_states
    
    

