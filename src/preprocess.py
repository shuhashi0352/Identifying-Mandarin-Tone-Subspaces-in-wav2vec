from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch
from tqdm import tqdm

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
    
    

