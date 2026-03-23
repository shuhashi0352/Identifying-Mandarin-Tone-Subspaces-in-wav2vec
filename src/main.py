from pathlib import Path
import yaml
from data import convert_wav, build_metadata

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_main(cfg):
    wav_dir = convert_wav(cfg)
    build_metadata(cfg, wav_dir)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)
    run_main(cfg)