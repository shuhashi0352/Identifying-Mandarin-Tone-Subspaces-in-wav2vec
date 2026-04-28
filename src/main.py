from pathlib import Path
import yaml
import subprocess
from audio_preprocess import convert_wav, build_metadata, build_mfa_lab
from create_dict import create_dict
from f0_extraction import extract_f0
from f0_standardization import speaker_zscore
from f0_interpolate import resample_f0
from speaker_stratification import run_stratification
from preprocess import create_train_test
from pca import pca_classification, pca_classification_aggregate, run_hidden_pca, run_multi_layer_pca, run_pca_intervention, run_pca_intervention_analysis
from probing import extract_hidden_states, run_split_hs, run_layerwise_probe
from tone_supervised import run_supervized
from train_das import run_das_best_layer, run_das_layer_sweep
from visualization import run_pca_visuals, run_pca_intervention_confusion_matrices

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_main(cfg):
    """
    from audio_preprocess import convert_wav, build_metadata, build_mfa_lab
    """
    # Convert mp3 to wav
    wav_dir = convert_wav(cfg)
    # Make a csv file for preprocessing and metadata
    df = build_metadata(cfg, wav_dir)
    # Make lab files corresponding to wav files for the forced alignment
    build_mfa_lab(df)

    """
    from create_dict import create_dict
    """
    # Create a pinyin dictionary for the MFA alignment
    create_dict(cfg)

    """
    MFA alignment via bash
    """
    # subprocess.run(["bash", "mfa.sh"], check=True)

    """
    from f0_extraction import extract_f0
    """
    # F0 extraction
    f0_results = extract_f0(cfg, time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)

    """
    from f0_standardization import speaker_zscore
    """
    # F0 standardization
    z_score, speaker_stats = speaker_zscore(f0_results)

    """
    from f0_interpolate import resample_f0
    """
    # F0 resampling (into the fixed num of frames: 20 frames as default)
    resampled = resample_f0(z_score)

    """
    from speaker_stratification import run_stratification
    """
    # Data split (speaker stratification)
    train_df, test_df = run_stratification(cfg, resampled)

    """
    from preprocess import create_train_test
    """
    # Match resampled items to the split
    re_x_train, re_y_train, re_x_test, re_y_test = create_train_test(resampled, train_df, test_df)

    """
    from pca import pca_classification, pca_classification_aggregate
    """
    # PCA using F0 exclusively
    # pca_classification(x_train, y_train, x_test, y_test)
    # Multiple results at once (num of components being k)
    # pca_classification_aggregate(re_x_train, re_y_train, re_x_test, re_y_test)

    """
    from probing import extract_hidden_states, run_split_hs, run_layerwise_probe
    """
    # Hidden states extraction
    hs_results = extract_hidden_states(cfg, resampled)
    # Hidden states split
    x_train_l0, y_train, x_test_l0, y_test, train_items, test_items = run_split_hs(hs_results, train_df, test_df)
    # Probing using lg
    run_layerwise_probe(train_items, test_items)

    """
    from pca import run_hidden_pca
    """
    # PCA probing
    # hs_pca_results = run_hidden_pca(train_items, test_items)
    # print(hs_pca_results)
    pca_df = run_multi_layer_pca(train_items, test_items)

    """
    from visualization import run_pca_visuals
    """
    # compact_df = run_pca_visuals(pca_df)

    """
    from pca import run_pca_intervention, run_pca_intervention_analysis
    """
    # PCA intervention 
    all_pc_df = run_pca_intervention(train_items, test_items)

    # PCA intervention analysis/visuals
    # run_pca_intervention_analysis(all_pc_df)

    """
    from visualization import run_pca_intervention_confusion_matrices
    """    
    # PCA intervention visuals
    # run_pca_intervention_confusion_matrices(train_items, test_items, all_pc_df, layer_idx=6)

    """
    from train_das import run_das
    """
    # DAS for a single (best) layer
    das_metrics = run_das_best_layer(train_items, test_items, cfg, layer_idx=6, k=8)
    print(das_metrics)
    # DAS for multiple layers
    # das_results = run_das_layer_sweep(train_items, test_items, cfg)
    # print(das_results)


    


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"
    cfg = load_yaml(CONFIG_PATH)
    run_main(cfg)