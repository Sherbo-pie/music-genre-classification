# =====================================================
# GTZAN FEATURE EXTRACTION SCRIPT (ERROR-SAFE VERSION)
# - Skips corrupted WAV files
# - Uses soundfile + scipy + python_speech_features
# - Saves MFCC features to gtzan_features.csv
# =====================================================

import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from python_speech_features import mfcc
import pandas as pd

# ----------------------------
# 1. AUDIO PREPROCESSING
# ----------------------------
def preprocess_audio(path, target_sr=22050):
    """
    Loads audio, converts to mono, resamples, normalizes.
    """
    audio, sr = sf.read(path, dtype='float32')

    # Convert stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down)
        sr = target_sr

    # Normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-12)

    return audio, sr

# ----------------------------
# 2. EXTRACT MFCC FEATURES
# ----------------------------
def extract_features(audio, sr, n_mfcc=13):
    """
    Computes MFCCs and returns mean + std (26 features).
    """
    mfcc_feat = mfcc(
        signal=audio,
        samplerate=sr,
        numcep=n_mfcc,
        winlen=0.032,
        winstep=0.016,
        nfilt=26,
        nfft=1024   # larger FFT to avoid warnings
    )

    mfcc_mean = np.mean(mfcc_feat, axis=0)
    mfcc_std = np.std(mfcc_feat, axis=0)

    return np.hstack([mfcc_mean, mfcc_std])  # 26 features

# ----------------------------
# 3. PROCESS ALL FILES IN GTZAN
# ----------------------------
def process_gtzan(root_folder):
    genres = sorted(os.listdir(root_folder))
    all_data = []

    for genre in genres:
        genre_path = os.path.join(root_folder, genre)
        wav_files = glob.glob(os.path.join(genre_path, "*.wav"))

        print(f"Processing {genre} ({len(wav_files)} files)...")

        for fpath in wav_files:
            try:
                audio, sr = preprocess_audio(fpath)
            except Exception as e:
                print("Skipping corrupted file:", fpath)
                continue

            feats = extract_features(audio, sr)

            # Store features + genre label
            row = list(feats) + [genre]
            all_data.append(row)

    # Column names
    columns = (
        [f"mfcc_mean_{i}" for i in range(13)] +
        [f"mfcc_std_{i}" for i in range(13)] +
        ["genre"]
    )

    df = pd.DataFrame(all_data, columns=columns)
    return df

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    root = r"C:\Users\Shirsha Pattanaik\Downloads\archive\Data\genres_original"
    df = process_gtzan(root)

    df.to_csv("gtzan_features.csv", index=False)
    print("\n✔ Feature extraction complete!")
    print("✔ Saved CSV: gtzan_features.csv")
