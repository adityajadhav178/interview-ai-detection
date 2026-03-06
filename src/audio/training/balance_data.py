import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

def handle_imbalance(features_csv, output_csv="data/processed/features_balanced.csv"):
    """
    Step 7: Handle Class Imbalance
    1. Check class distribution and visualize
    2. Apply SMOTE to balance the classes
    """
    # 1. Load data
    try:
        df = pd.read_csv(features_csv)
    except Exception as e:
        print(f"Error reading {features_csv}: {e}")
        return
        
    if "label" not in df.columns:
        print("Error: 'label' column not found in dataset.")
        return
        
    print(f"Original Dataset Shape: {df.shape}")
    
    # Analyze original distribution
    label_counts = df["label"].value_counts().sort_index()
    print("\n--- Original Class Distribution ---")
    for cls, count in label_counts.items():
        print(f"Class {cls}: {count} samples")
        
    # Visualize original distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Original Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    
    # Optional: Compute class_weight parameter as an alternative to SMOTE
    # This can be passed to models like RandomForestClassifier(class_weight='balanced')
    # or XGBClassifier/LightGBM
    classes = np.unique(df["label"])
    weights = compute_class_weight('balanced', classes=classes, y=df["label"])
    class_weights_dict = dict(zip(classes, weights))
    
    print("\n--- Alternative: Class Weights ---")
    print("If you prefer NOT to use SMOTE, use these class weights in your model:")
    for cls, weight in class_weights_dict.items():
        print(f"Class {cls}: weight = {weight:.4f}")
        
    # Prepare data for SMOTE
    # Drop patient_id as it's just an identifier, save it if needed
    if "patient_id" in df.columns:
        X = df.drop(["label", "patient_id"], axis=1)
        patient_ids = df["patient_id"]
    else:
        X = df.drop("label", axis=1)
        
    y = df["label"]
    
    # 2. Apply SMOTE
    print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique)...")
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"SMOTE Error: {e}")
        print("Note: SMOTE requires at least 6 samples per class by default (k_neighbors=5).")
        print("If you have very few samples, you might need to adjust k_neighbors or collect more data.")
        return
        
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    print("\n--- Balanced Class Distribution ---")
    for cls, count in resampled_counts.items():
        print(f"Class {cls}: {count} samples (Resampled)")
        
    # Visualize balanced distribution
    plt.subplot(1, 2, 2)
    sns.barplot(x=resampled_counts.index, y=resampled_counts.values)
    plt.title("Balanced Class Distribution (After SMOTE)")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plot_path = Path("data/processed/class_distribution.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    print(f"\nSaved distribution plot to {plot_path}")
    
    # Combine back to dataframe
    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_df.insert(0, "label", y_resampled)
    # Note: patient_id is lost for synthetic samples. 
    # For synthetic records, we can assign 'SYNTHETIC' or similar
    
    # Create patient_ids for the new dataframe
    # The original samples keep their IDs, new ones get 'SYNTH_xx'
    original_len = len(patient_ids)
    new_len = len(y_resampled)
    
    if "patient_id" in df.columns:
        new_ids = ["SYNTHETIC_" + str(i) for i in range(new_len - original_len)]
        all_ids = list(patient_ids) + new_ids
        balanced_df.insert(0, "patient_id", all_ids)
        
    # Save the balanced dataset
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(out_path, index=False)
    
    print(f"\nSuccessfully balanced dataset! New Shape: {balanced_df.shape}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance dataset using SMOTE and compute class weights")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to features.csv")
    parser.add_argument("--output_csv", type=str, default="data/processed/features_balanced.csv", help="Path to save balanced dataset")
    
    args = parser.parse_args()
    
    handle_imbalance(args.input_csv, args.output_csv)
