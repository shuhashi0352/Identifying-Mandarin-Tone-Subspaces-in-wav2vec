from pathlib import Path
from typing import Set, List, Tuple
import pandas as pd

from pinyin_to_ipa import pinyin_to_ipa

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "data/metadata.csv"
TOKEN_COL = "sound"          # default: segmental syllable only
WAV_COL = "wav_path"
DICTIONARY_PATH = "data/mfa_dictionary.txt"
WRITE_LAB = True

# dianr can be transcribed as [tjɐʵ] or [tjɐɻ]
# either we consider ɐʵ as one or two phones
# separating phones can make it harder to find the start and stop times
SPLIT_PHONEMES = []  # e.g. [("ʵ", " ɻ"), ("˞", " ɻ"), ("ɚ", "ə ɻ"), ("aə", "a")]

# Dictionary for erhua transformations
# https://en.wikipedia.org/wiki/Erhua#Standard_rules
ERHUA_SUFFIX_TO_IPA = {
    "uangr": [["w", "ɑ̃ʵ"]],
    "iangr": [["j", "ɑ̃ʵ"]],
    "iongr": [["j", "ʊ̃ʵ"]],
    "vanr": [["ɥ", "ɐʵ"]],
    "uair": [["w", "ɐʵ"]],
    "ianr": [["j", "ɐʵ"]],
    "iaor": [["j", "ɑu̯˞"]],
    "uanr": [["w", "ɐʵ"]],
    "engr": [["ɤ̃ʵ"]],
    "angr": [["ɑ̃ʵ"]],
    "ongr": [["w", "ɤ̃ʵ"], ["ʊ̃˞"]],
    "ingr": [["j", "ɤ̃ʵ"]],
    "ver": [["ɥ", "œʵ"]],
    "uar": [["w", "äʵ"], ["w", "ɐʵ"]],
    "uor": [["w", "ɔʵ"]],
    "air": [["ɐʵ"]],
    "eir": [["ɚ"]],
    "aor": [["ɑu̯˞"]],
    "our": [["ou̯˞"]],
    "anr": [["ɐʵ"]],
    "enr": [["ɚ"]],
    "iar": [["j", "äʵ"], ["j", "ɐʵ"]],
    "ier": [["j", "ɛʵ"]],
    "iur": [["j", "ou̯ʵ"]],
    "inr": [["j", "ɚ"]],
    "uir": [["w", "ɚ"]],
    "unr": [["w", "ɚ"]],
    "vnr": [["ɥ", "ɚ"]],
    "ar": [["äʵ"], ["ɐʵ"]],
    "or": [["ɔʵ"]],
    "er": [["ɤʵ"]],
    "ur": [["u˞"]],
    "vr": [["ɥ", "ɚ"]],
    "ir": [["ɚ"]],
}


def apply_erhua(pinyin: str, ipa: List[str]) -> List[List[str]]:
    """
    Apply erhua transformation to the given IPA representation.
    Returns a list of possible IPA outputs.
    """
    ipa = ipa.copy()

    if ipa and ipa[-1] in {"ŋ", "n"}:
        ipa = ipa[:-1]

    if len(ipa) > 1:
        ipa = ipa[:-1]

    results = []
    for pinyin_ending, ipa_endings in ERHUA_SUFFIX_TO_IPA.items():
        if pinyin.endswith(pinyin_ending) and pinyin != pinyin_ending:
            for ipa_ending in ipa_endings:
                new_ipa = ipa.copy()
                new_ending = ipa_ending.copy()

                if new_ipa and new_ending and new_ipa[-1] == new_ending[0]:
                    new_ending = new_ending[1:]
                elif new_ipa and new_ending and new_ipa[-1] + "˞" == new_ending[0]:
                    new_ipa = []

                results.append(new_ipa + new_ending)
            break
    return results

def convert_pinyin_to_ipa(token: str) -> List[List[str]]:
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


def load_tokens_from_csv(csv_path: str | Path, token_col: str) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load metadata.csv and collect unique transcript tokens.
    """
    df = pd.read_csv(csv_path)

    required_cols = {token_col, WAV_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.copy()
    df[token_col] = df[token_col].astype(str).str.strip().str.lower()
    df[WAV_COL] = df[WAV_COL].astype(str).str.strip()

    # Remove missing/empty tokens and malformed wav paths
    df = df[(df[token_col] != "") & (df[token_col] != "nan")]
    df = df[(df[WAV_COL] != "") & (df[WAV_COL] != "nan")]

    tokens = set(df[token_col].unique())
    print(f"Number of unique tokens in '{token_col}': {len(tokens)}")
    return df, tokens


def write_lab_files(df: pd.DataFrame, token_col: str) -> None:
    """
    Create one .lab file next to each .wav file.
    For your dataset, each row is a single-syllable token.
    """
    for _, row in df.iterrows():
        wav_path = Path(row[WAV_COL])
        transcript = row[token_col]

        lab_path = wav_path.with_suffix(".lab")
        lab_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(transcript)


def generate_dictionary_entries(tokens: Set[str]) -> List[Tuple[str, str]]:
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


def write_dictionary(entries: List[Tuple[str, str]], filename: str | Path) -> None:
    """
    Write MFA-style dictionary:
        token<TAB>ipa ipa ipa
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for token, ipa in entries:
            f.write(f"{token}\t{ipa}\n")


def create_dict():
    df, tokens = load_tokens_from_csv(CSV_PATH, TOKEN_COL)

    if WRITE_LAB:
        print("Writing .lab files...")
        write_lab_files(df, TOKEN_COL)

    print("Generating dictionary...")
    entries = generate_dictionary_entries(tokens)
    write_dictionary(entries, DICTIONARY_PATH)

    print(f"Dictionary written to: {DICTIONARY_PATH}")