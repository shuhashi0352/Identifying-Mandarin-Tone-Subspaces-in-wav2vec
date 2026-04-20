from pathlib import Path
from typing import Set, List, Tuple
import pandas as pd
import yaml
from pinyin_to_ipa import pinyin_to_ipa

def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Skip writing .lab files if they already exist
WRITE_LAB = True

# separating phones can make it harder to find the start and stop times
SPLIT_PHONEMES = [] 

def convert_pinyin_to_ipa(token):
    """
    Convert one pinyin token to a simplified IPA output for MFA.

    Simplifications:
    - keep only one pronunciation per token
    - map syllabic consonants to i
    - split diphthongs into simpler sequences
    - force a single pronunciation for er
    - collapse h/x alternation by choosing one consistently
    """
    token = token.strip().lower()

    # Force one pronunciation for er
    if token == "er":
        return [["ɚ"]]

    ipa_outputs: List[List[str]] = []

    for ipa in pinyin_to_ipa(token):
        ipa = list(ipa)

        # Special case for yo
        if token == "yo" and ipa == ["w", "o"]:
            ipa = ["j", "ɔ"]

        simplified = []
        for ph in ipa:
            # Simplify syllabic consonants
            if ph in {"ɻ̩", "ʐ̩", "ɹ̩", "z̩"}:
                simplified.append("i")

            # Simplify diphthongs
            elif ph == "ai̯":
                simplified.extend(["a", "i"])
            elif ph == "au̯":
                simplified.extend(["a", "u"])
            elif ph == "ou̯":
                simplified.extend(["o", "u"])
            elif ph == "ei̯":
                simplified.extend(["e", "i"])

            # Simplify h/x alternation: choose x
            elif ph == "h":
                simplified.append("x")

            else:
                simplified.append(ph)

        # keep only one pronunciation per token
        if simplified not in ipa_outputs:
            ipa_outputs.append(simplified)

    # return only the first pronunciation
    return ipa_outputs[:1]

def load_tokens_from_csv(csv_path, token_col, wav_col):
    """
    Load metadata.csv and collect unique transcript tokens.
    """
    df = pd.read_csv(csv_path)

    required_cols = {token_col, wav_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.copy()
    df[token_col] = df[token_col].astype(str).str.strip().str.lower()
    df[wav_col] = df[wav_col].astype(str).str.strip()

    # Remove missing/empty tokens and malformed wav paths
    df = df[(df[token_col] != "") & (df[token_col] != "nan")]
    df = df[(df[wav_col] != "") & (df[wav_col] != "nan")]

    tokens = set(df[token_col].unique())
    print(f"Number of unique tokens in '{token_col}': {len(tokens)}")
    return df, tokens

def write_lab_files(df, token_col, wav_col):
    """
    Create one .lab file next to each .wav file.
    For your dataset, each row is a single-syllable token.
    """
    for _, row in df.iterrows():
        wav_path = Path(row[wav_col])
        transcript = row[token_col]

        lab_path = wav_path.with_suffix(".lab")
        lab_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(transcript)

def generate_dictionary_entries(tokens):
    """
    Build dictionary entries: token -> IPA sequence.
    """
    entries: List[Tuple[str, str]] = []
    failed_tokens: List[str] = []

    for token in sorted(tokens):
        try:
            ipas = convert_pinyin_to_ipa(token)
        except Exception:
            failed_tokens.append(token)
            continue

        if not ipas:
            failed_tokens.append(token)
            continue

        for ipa in ipas:
            str_ipa = " ".join(ipa)
            for s0, s1 in SPLIT_PHONEMES:
                str_ipa = str_ipa.replace(s0, s1)
            entries.append((token, str_ipa))

    entries = sorted(set(entries))

    phone_inventory = sorted({ph for _, pron in entries for ph in pron.split()})
    print(f"Number of dictionary entries: {len(entries)}")
    print(f"Number of unique phones: {len(phone_inventory)}")

    if failed_tokens:
        print(f"WARNING: {len(failed_tokens)} token(s) could not be converted.")
        print("First few failed tokens:", failed_tokens[:20])

    return entries

def write_dictionary(entries, filename):
    """
    Write MFA-style dictionary:
        token<TAB>ipa ipa ipa
    """
    print(filename)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for token, ipa in entries:
            f.write(f"{token}\t{ipa}\n")

def create_dict(cfg):
    DATA_PATH = cfg["data"]
    CSV_PATH = DATA_PATH["metadata_path"]
    TOKEN_COL = DATA_PATH["sound_col"]
    WAV_COL = DATA_PATH["audio_col"]
    DICT_PATH = DATA_PATH["mfa_dict_path"]

    df, tokens = load_tokens_from_csv(CSV_PATH, TOKEN_COL, WAV_COL)

    if WRITE_LAB:
        print("Writing .lab files...")
        write_lab_files(df, TOKEN_COL, WAV_COL)

    print("Generating dictionary...")
    entries = generate_dictionary_entries(tokens)
    write_dictionary(entries, DICT_PATH)

    print(f"Dictionary written to: {DICT_PATH}")