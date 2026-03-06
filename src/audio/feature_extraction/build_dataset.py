import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

# Import the pipeline functions we built in earlier steps
from src.audio.preprocessing.audio_cleaner import preprocess_audio
from src.audio.feature_extraction.extract_features import extract_audio_features

# Label mapping derived from folder names (no labels.csv needed)
CONDITION_LABEL_MAP = {
    "normal":     0,
    "depression": 1,
    "anxiety":    2,
    "adhd":       3,
    "ocd":        4,
}

def aggregate_features(features_list):
    """
    Step 5: Aggregate features from multiple answers.
    Takes a list of feature dicts and computes mean, std, min, max, median.
    Output: 1 flat dictionary with 100 features (20 x 5 aggregations).
    """
    if not features_list:
        return {}

    df = pd.DataFrame(features_list)
    aggregated = {}
    for col in df.columns:
        aggregated[f"{col}_mean"]   = df[col].mean()
        aggregated[f"{col}_std"]    = df[col].std()
        aggregated[f"{col}_min"]    = df[col].min()
        aggregated[f"{col}_max"]    = df[col].max()
        aggregated[f"{col}_median"] = df[col].median()

    return aggregated

def build_dataset(data_dir, output_csv="data/processed/features.csv"):
    """
    Scans the folder structure under data_dir and builds features.csv.
    
    Expected folder structure:
        data_dir/
            depression/
                patient_001/
                    answer_1.wav
                    answer_2.wav
                    ...
                patient_002/
                    ...
            normal/
                patient_010/
                    ...
            anxiety/
                ...

    Labels are derived automatically from the condition folder name.
    No labels.csv required.
    """
    data_path = Path(data_dir)
    final_data = []

    # Iterate over each condition folder (depression, normal, etc.)
    condition_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if not condition_dirs:
        print(f"No subfolders found in {data_dir}.")
        return

    for condition_dir in sorted(condition_dirs):
        condition_name = condition_dir.name.lower()

        if condition_name not in CONDITION_LABEL_MAP:
            print(f"Skipping unknown condition folder: '{condition_name}'. (Expected one of: {list(CONDITION_LABEL_MAP.keys())})")
            continue

        label = CONDITION_LABEL_MAP[condition_name]
        patient_dirs = [d for d in condition_dir.iterdir() if d.is_dir()]
        print(f"\nProcessing condition: '{condition_name}' (label={label}) — {len(patient_dirs)} patients found.")

        for patient_dir in tqdm(patient_dirs, desc=f"  {condition_name}"):
            patient_id = patient_dir.name
            wav_files  = list(patient_dir.glob("*.wav"))

            if not wav_files:
                print(f"\n  Warning: No .wav files in {patient_dir}. Skipping.")
                continue

            patient_features = []
            for wav_file in wav_files:
                try:
                    clean_audio, sr = preprocess_audio(str(wav_file))
                    if clean_audio is not None and len(clean_audio) > 0:
                        feats = extract_audio_features(clean_audio, sr)
                        if feats:
                            patient_features.append(feats)
                except Exception as e:
                    print(f"\n  Error processing {wav_file.name}: {e}")

            if patient_features:
                aggregated = aggregate_features(patient_features)
                row = {"patient_id": patient_id, "condition": condition_name, "label": label}
                row.update(aggregated)
                final_data.append(row)
            else:
                print(f"\n  Warning: No features extracted for {patient_id}.")

    if not final_data:
        print("\nNo data to save! Place patient wav files inside condition subfolders.")
        return

    out_df   = pd.DataFrame(final_data)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\n✅ Dataset built! Shape: {out_df.shape}")
    print(f"   Saved to: {out_path}")
    print("\nClass distribution:")
    print(out_df["condition"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build features.csv from folder-structured audio data. Labels are auto-derived from folder names."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/audio",
        help="Root audio folder containing condition subfolders (e.g., data/raw/audio)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/processed/features.csv",
        help="Path to save the output features CSV"
    )

    args = parser.parse_args()
    build_dataset(args.data_dir, args.output_csv)
