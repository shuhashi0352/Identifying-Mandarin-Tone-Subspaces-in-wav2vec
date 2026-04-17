##################################
###### MFA Forced Alignment ######
##################################

# 1. Prepare a dictionary to map sound to IPA (manual from main.py)
# 2. Train an acoustic model + alignment (manual)
# 3. Textgrid files are given

##################################

# Immidiate kill if fails; handling errors
set -euo pipefail

# conda create -n mfa -c conda-forge python=3.11 montreal-forced-aligner=3.3.9 
# - to forge an mfa conda environment
# (Skip this process if you've already created the environment)

# Load conda's shell into bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mfa

# Install MFA + compiled dependencies only if missing
# fstcompile is an OpenFst binary MFA/Kaldi needs for compiling graph resources
command -v fstcompile >/dev/null 2>&1 || \

# Install MFT in the specified conda environment
conda install -y -c conda-forge montreal-forced-aligner=3.3.9 kalpy kaldi

# CAUTION: MFA doesn't necessarily live in this path (reproducability)
MFA_BIN="$HOME/miniconda3/envs/mfa/bin/mfa"

# Install packages required for alignment
python -c "import spacy_pkuseg, dragonmapper, hanziconv" 2>/dev/null || \
python -m pip install spacy-pkuseg dragonmapper hanziconv

mkdir -p data/mfa_textgrids

# Train the acoustic model
# Temporarily treat the corpus as single-speaker because the current folder layout
# does not expose speaker structure to MFA (It won't hurt in consonant removal)
"$MFA_BIN" train \
  --clean \
  --single_speaker \
  --phone_set IPA \
  --output_directory data/mfa_textgrids \
  data/tone_perfect_wav \
  data/mfa_dictionary.txt \
  data/mfa_acoustic_model.zip

# data/mfa_textgrids -> output from the alignment
# data/tone_perfect_wav -> audio files
# data/mfa_dictionary.txt -> character-to-phone dictionary (.txt)
# mfa_acoustic_model.zip -> acoustic model

#################################

##### Utilities #####

# which mfa
# - to check if the path aligns with MFA_BIN

# chmod +x mfa.sh
# - to make this file executable

# bash mfa.sh
# - to run this bash script