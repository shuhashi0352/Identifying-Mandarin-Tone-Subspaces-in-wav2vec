from pathlib import Path
import yaml
import subprocess
from data import convert_wav, build_metadata, build_mfa_lab
from create_dict import create_dict
from preprocess import hs_extraction

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_main(cfg):
    # Convert mp3 to wav
    wav_dir = convert_wav(cfg)

    # Make a csv file for preprocessing and metadata
    df = build_metadata(cfg, wav_dir)

    # Make lab files corresponding to wav files for the forced alignment
    build_mfa_lab(df)

    # Create a pinyin dictionary for the MFA alignment
    create_dict()

    # MFA alignment via bash
    subprocess.run(["bash", "mfa.sh"], check=True)

    # Hidden states extraction
    # hidden_states = hs_extraction(df)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)
    run_main(cfg)