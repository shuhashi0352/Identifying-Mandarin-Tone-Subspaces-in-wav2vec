import numpy as np
from collections import defaultdict

def speaker_zscore(results):
    """
    results: list of dicts from extract_f0(...)
    each item contains:
        wav_file
        f0
    returns:
        updated results with z-scored f0
    """

    speaker_values = defaultdict(list)

    # collect voiced F0 values per speaker
    for item in results:
        wav_name = item["wav_file"].stem
        speaker = wav_name.split("_")[1]   # FV1, MV2, etc.

        vals = item["f0"][~np.isnan(item["f0"])]
        if len(vals) > 0:
            speaker_values[speaker].extend(vals.tolist())

    # compute speaker stats
    stats = {}
    for spk, vals in speaker_values.items():
        vals = np.array(vals)
        stats[spk] = {
            "mean": vals.mean(),
            "std": vals.std(ddof=0)
        }

    # transform
    z_results = []

    for item in results:
        wav_name = item["wav_file"].stem
        speaker = wav_name.split("_")[1]

        mu = stats[speaker]["mean"]
        sd = stats[speaker]["std"]

        f0 = item["f0"].copy()
        zf0 = np.where(np.isnan(f0), np.nan, (f0 - mu) / sd)

        new_item = item.copy()
        new_item["speaker"] = speaker
        new_item["f0_z"] = zf0
        z_results.append(new_item)
    
    # print(z_results)

    return z_results, stats