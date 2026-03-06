import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

def get_feature_category(feature_name):
    """Map feature name to one of 5 categories for analysis"""
    name = feature_name.lower()
    if 'mfcc' in name:
        return 'MFCC'
    elif 'pitch' in name:
        return 'Pitch'
    elif 'energy' in name or 'rms' in name:
        return 'Energy'
    elif any(term in name for term in ['speech_rate', 'pause', 'zero_crossing']):
        return 'Speech Patterns'
    elif 'jitter' in name or 'shimmer' in name:
        return 'Voice Quality'
    else:
        return 'Other'

def analyze_feature_importance(model, feature_names, model_name, output_dir):
    """Extract, plot top 20 features, and group by category"""
    if not hasattr(model, 'feature_importances_'):
        return
        
    importances = model.feature_importances_
    
    # Store in dataframe
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Category': [get_feature_category(f) for f in feature_names]
    })
    
    # Sort and get top 20
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    top_20 = feat_imp.head(20)
    
    # 1. Plot Top 20 Features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_20, hue='Category', dodge=False)
    plt.title(f"Top 20 Features ({model_name})")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_top_features.png")
    plt.close()
    
    # 2. Group by Category contribution
    cat_imp = feat_imp.groupby('Category')['Importance'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cat_imp.values, y=cat_imp.index, palette='viridis')
    plt.title(f"Feature Importance by Category ({model_name})")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_category_importance.png")
    plt.close()
    
    print(f"\n--- Feature Importance Summary ({model_name}) ---")
    print(f"Top 5 Individual Features:")
    for _, row in top_20.head(5).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")
        
    print("\nTotal Contribution by Category:")
    for cat, imp in cat_imp.items():
        print(f"  - {cat}: {imp:.4f} ({(imp/cat_imp.sum())*100:.1f}%)")
        
    most_important = cat_imp.index[0]
    print(f"\nThe '{most_important}' category contributes MOST to prediction in this model.")

def train_and_evaluate(input_csv, output_model_dir, eval_dir):
    """
    Step 8: Train Models
    Step 9: Feature Importance
    """
    out_dir = Path(output_model_dir)
    ev_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading balanced dataset from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Dataset {input_csv} not found.")
        return
        
    # Drop identifiers and extract labels
    if "patient_id" in df.columns:
        X = df.drop(["label", "patient_id"], axis=1)
    else:
        X = df.drop("label", axis=1)
        
    y = df["label"]
    feature_names = X.columns.tolist()
    
    # 1. Train-test split (80/20 stratification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Apply StandardScaler normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    joblib.dump(scaler, out_dir / "standard_scaler.joblib")
    print(f"Saved StandardScaler to {out_dir / 'standard_scaler.joblib'}")
    
    # 3. Define Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    }
    
    # Remap labels to 0-N sequentially for XGBoost/LightGBM compatibility if they aren't already
    UNIQUE_LABELS = sorted(list(np.unique(y_train)))
    label_mapping = {val: idx for idx, val in enumerate(UNIQUE_LABELS)}
    
    y_train_mapped = np.array([label_mapping[val] for val in y_train])
    y_test_mapped = np.array([label_mapping[val] for val in y_test])
    
    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    # 5-Fold Cross validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Train, Evaluate, Print
    for name, model in models.items():
        print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
        
        # Cross Validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train_mapped, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Fit on full 80% train set
        model.fit(X_train_scaled, y_train_mapped)
        
        # Evaluate on 20% test set
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test_mapped, y_pred)
        
        print(f"Test Set Accuracy: {test_acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test_mapped, y_pred, target_names=[str(l) for l in UNIQUE_LABELS]))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_mapped, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=UNIQUE_LABELS, yticklabels=UNIQUE_LABELS)
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(ev_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.close()
        
        # Keep track of best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_name = name
            best_model = model
            
        # Step 9: Feature Importance
        if hasattr(model, 'feature_importances_'):
            analyze_feature_importance(model, feature_names, name, ev_dir)
            
    # 5. Save best model
    print(f"\n{'*'*50}")
    print(f"Best Model: {best_model_name} with Accuracy {best_accuracy:.4f}")
    
    model_path = out_dir / "best_audio_model.joblib"
    # Also save the label mapping so inference knows what 0,1,2,3 means
    joblib.dump({"model": best_model, "label_mapping": {v: k for k, v in label_mapping.items()}}, model_path)
    print(f"Saved Best Model to {model_path}")
    print(f"{'*'*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train evaluation pipeline for audio features")
    parser.add_argument("--input_csv", type=str, default="data/processed/features_balanced.csv")
    parser.add_argument("--model_dir", type=str, default="models/audio")
    parser.add_argument("--eval_dir", type=str, default="evaluation/audio")
    
    args = parser.parse_args()
    train_and_evaluate(args.input_csv, args.model_dir, args.eval_dir)
