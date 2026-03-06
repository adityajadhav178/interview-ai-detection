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

def build_dataset(data_dir, output_dir="data/processed/categories"):
    """
    Scans the folder structure under data_dir and builds a CSV for each category.
    
    Supports both nested folders (patient_001/*.wav) and flat folders (*.wav).
    
    Labels are derived automatically from the condition folder name.
    """
    data_path = Path(data_dir)
    out_dir_path = Path(output_dir)
    
    # Iterate over each condition folder (depression, normal, etc.)
    condition_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if not condition_dirs:
        print(f"No subfolders found in {data_dir}.")
        return

    out_dir_path.mkdir(parents=True, exist_ok=True)
    total_samples = 0

    for condition_dir in sorted(condition_dirs):
        condition_name = condition_dir.name.lower()

        if condition_name not in CONDITION_LABEL_MAP:
            print(f"Skipping unknown condition folder: '{condition_name}'. (Expected one of: {list(CONDITION_LABEL_MAP.keys())})")
            continue

        label = CONDITION_LABEL_MAP[condition_name]
        patient_dirs = [d for d in condition_dir.iterdir() if d.is_dir()]
        condition_data = []

        if patient_dirs:
            print(f"\nProcessing condition: '{condition_name}' (label={label}) — {len(patient_dirs)} patients found.")
            for patient_dir in tqdm(patient_dirs, desc=f"  {condition_name}"):
                patient_id = patient_dir.name
                wav_files  = list(patient_dir.glob("*.wav"))

                if not wav_files:
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
                    condition_data.append(row)
        else:
            wav_files = list(condition_dir.glob("*.wav"))
            print(f"\nProcessing condition: '{condition_name}' (label={label}) — {len(wav_files)} flat wav files found.")
            for wav_file in tqdm(wav_files, desc=f"  {condition_name}"):
                patient_id = wav_file.stem
                try:
                    clean_audio, sr = preprocess_audio(str(wav_file))
                    if clean_audio is not None and len(clean_audio) > 0:
                        feats = extract_audio_features(clean_audio, sr)
                        if feats:
                            # Aggregate still runs to ensure format matches (mean/min/max/std/median)
                            aggregated = aggregate_features([feats])
                            row = {"patient_id": patient_id, "condition": condition_name, "label": label}
                            row.update(aggregated)
                            condition_data.append(row)
                except Exception as e:
                    print(f"\n  Error processing {wav_file.name}: {e}")

        if condition_data:
            out_df = pd.DataFrame(condition_data)
            category_out_path = out_dir_path / f"features_{condition_name}.csv"
            out_df.to_csv(category_out_path, index=False)
            total_samples += len(condition_data)
            print(f"  -> Saved {len(condition_data)} samples to {category_out_path}")
        else:
            print(f"\n  Warning: No features extracted for condition {condition_name}.")

    if total_samples > 0:
        print(f"\n✅ All available datasets built! Extracted features for {total_samples} total patients/files.")
        print(f"   Outputs saved in: {out_dir_path}")
    else:
        print("\nNo data to save! Place patient wav files inside condition subfolders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build separate CSV features for each condition. Labels derived from folder names."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/audio",
        help="Root audio folder containing condition subfolders (e.g., data/raw/audio)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/categories",
        help="Directory to save the per-category output CSV files"
    )

    args = parser.parse_args()
    build_dataset(args.data_dir, args.output_dir)
