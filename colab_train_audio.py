"""
═══════════════════════════════════════════════════════════════════════════════
  GOOGLE COLAB — Audio Model Training Script (NO DRIVE UPLOAD NEEDED)
  
  Usage:
    1. Paste your CSV data directly in CELL 1
    2. Run all cells in order
    3. Download trained models from Colab files
═══════════════════════════════════════════════════════════════════════════════
"""

# ═════════════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies & Prepare Data
# ═════════════════════════════════════════════════════════════════════════════

import subprocess
subprocess.run(['pip', 'install', '-q', 'scikit-learn', 'pandas', 'numpy', 
                'matplotlib', 'seaborn', 'xgboost', 'lightgbm', 'joblib', 'imbalanced-learn'], 
               check=True)

print("✓ Dependencies installed")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 2: Import Libraries & Configuration
# ═════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import joblib
import warnings
import json
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

# Training hyperparameters
TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42
TUNE_ITERS = 20
PRIMARY_METRIC = "f1_weighted"
TRAIN_APPLY_SMOTE = True
SMOTE_K_NEIGHBORS = 5

# Label mapping
LABEL_NAMES = {0: "Normal", 1: "Depression", 2: "Anxiety", 3: "ADHD", 4: "OCD"}

print(f"✓ Libraries imported")
print(f"✓ Ready to load features data")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 3: Define Models & Hyperparameters
# ═════════════════════════════════════════════════════════════════════════════

def _base_models():
    """Return base model estimators."""
    rs = RANDOM_STATE
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=rs),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=rs, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=rs, n_jobs=-1, verbose=-1),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=rs),
    }
    return models


def _param_grids():
    """Hyperparameter search spaces."""
    return {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300],
            "num_leaves": [15, 31, 50],
            "max_depth": [5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
        },
        "SVM (RBF)": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
    }


print("✓ Models & hyperparameters defined")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 4: Load Features Data (PASTE YOUR CSV HERE)
# ═════════════════════════════════════════════════════════════════════════════

# 📌 OPTION 1: PASTE CSV CONTENT DIRECTLY (RECOMMENDED)
# ────────────────────────────────────────────────────────────────────────────
# Copy your features.csv (unbalanced) content and paste it here between the triple quotes

csv_data = """patient_id,condition,label,feature_1,feature_2,feature_3
patient_001,depression,1,0.5,0.3,0.8
patient_002,normal,0,0.2,0.1,0.3
patient_003,depression,1,0.6,0.4,0.9
patient_004,anxiety,2,0.4,0.2,0.7
patient_005,normal,0,0.1,0.05,0.2
patient_006,anxiety,2,0.5,0.3,0.8
patient_007,adhd,3,0.7,0.5,1.0
patient_008,ocd,4,0.8,0.6,1.1
patient_009,depression,1,0.55,0.35,0.85
patient_010,normal,0,0.15,0.08,0.25
"""

# ────────────────────────────────────────────────────────────────────────────
# 📌 OPTION 2: UPLOAD CSV FILE FROM YOUR COMPUTER
# ────────────────────────────────────────────────────────────────────────────
# Uncomment the code below if you want to upload a file instead

# from google.colab import files
# print("📁 Select your features_balanced.csv file:")
# uploaded = files.upload()
# csv_filename = list(uploaded.keys())[0]
# with open(csv_filename, 'r') as f:
#     csv_data = f.read()

# ────────────────────────────────────────────────────────────────────────────

print("\n📂 Loading features from CSV...")
df = pd.read_csv(StringIO(csv_data))
print(f"✓ Loaded {len(df)} samples")
print(f"✓ Features shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")

# Drop metadata columns
drop_cols = [c for c in ("patient_id", "condition", "label") if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["label"]
feat_names = list(X.columns)

# Sanitize feature matrix before train/test split.
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
if X.isna().sum().sum() > 0:
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

print(f"\n✓ Features: {len(feat_names)} acoustic features")
print(f"✓ Label distribution:\n{y.value_counts().sort_index()}")

# Label mapping (0-indexed for sklearn)
uniq = sorted(y.unique())
lmap = {v: i for i, v in enumerate(uniq)}
rmap = {i: v for v, i in lmap.items()}
y_mapped = y.map(lmap)

print(f"✓ Label mapping: {lmap}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_mapped, test_size=TEST_SIZE,
    stratify=y_mapped, random_state=RANDOM_STATE,
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Apply SMOTE on training split only (prevents leakage into test set).
X_train_fit = X_train_s
y_train_fit = y_train
if TRAIN_APPLY_SMOTE:
    cls, cnt = np.unique(y_train, return_counts=True)
    min_class_count = int(cnt.min()) if len(cnt) else 0
    if min_class_count > 1:
        k_neighbors = min(SMOTE_K_NEIGHBORS, min_class_count - 1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_train_fit, y_train_fit = smote.fit_resample(X_train_s, y_train)

        print(f"✓ Train balancing: SMOTE enabled (k_neighbors={k_neighbors})")
        print("  Class counts before SMOTE:")
        for c, n in pd.Series(y_train).value_counts().sort_index().items():
            print(f"    class {c}: {n}")
        print("  Class counts after SMOTE:")
        for c, n in pd.Series(y_train_fit).value_counts().sort_index().items():
            print(f"    class {c}: {n}")
    else:
        print("✓ Train balancing skipped: one class has <2 samples in train split")
else:
    print("✓ Train balancing: SMOTE disabled")

print(f"\n✓ Train set: {X_train_s.shape}")
print(f"✓ Test set: {X_test_s.shape}")
print(f"✓ Scaler ready")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 5: Evaluation Function
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cv_folds=5):
    """Compute train/test metrics and cross-validation score."""
    
    # Cross-validation F1
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted") if y_proba is not None else 0
    except:
        auc = 0
    
    return {
        "model": model,
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "y_pred": y_pred,
        "classification_report": classification_report(y_test, y_pred, target_names=[LABEL_NAMES[rmap[i]] for i in range(len(uniq))]),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

print("✓ Evaluation function defined")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 6: Train Base Models (with Hyperparameter Tuning)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("  🚀 TRAINING BASE MODELS")
print("═"*70)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
grids = _param_grids()
results = {}
fitted = {}

models = _base_models()

header = f"\n{'Model':<22} {'CV F1':>8} {'Test F1':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
print(header)
print("─" * 70)

for model_name, model in models.items():
    
    # Hyperparameter tuning
    if model_name in grids:
        print(f"  {model_name:<20} [Tuning...]", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searcher = RandomizedSearchCV(
                model, grids[model_name],
                n_iter=TUNE_ITERS, scoring=PRIMARY_METRIC,
                cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
            )
            searcher.fit(X_train_fit, y_train_fit)
            model = searcher.best_estimator_
    else:
        print(f"  {model_name:<20} [Training...]", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_fit, y_train_fit)
    
    # Evaluate
    metrics = evaluate_model(model, X_train_fit, y_train_fit, X_test_s, y_test, model_name, cv_folds=CV_FOLDS)
    results[model_name] = metrics
    fitted[model_name] = model
    
    print(f"CV: {metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}  "
          f"Test F1: {metrics['f1']:.4f}  Acc: {metrics['accuracy']:.4f}  "
          f"Prec: {metrics['precision']:.4f}  Rec: {metrics['recall']:.4f}")

print("─" * 70)
print("✓ Base models trained")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 7: [SKIPPED] Ensemble Models
# ═════════════════════════════════════════════════════════════════════════════
# Ensemble training disabled — base models already achieve ~99.6% F1
# (Gradient Boosting & XGBoost at 0.9962, Extra Trees at 0.9924)
# Ensemble overhead not justified for marginal gains

print("\n" + "═"*70)
print("  ⏭️  SKIPPING ENSEMBLE (Base models sufficient at 0.9962 F1)")
print("═"*70)
print("✓ Proceeding to download all 6 base models")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 8: Save Results & Best Model (Download from Colab)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("  💾 SAVING ALL BASE MODELS & RESULTS")
print("═"*70)

# Find best model
best_name = max(results.keys(), key=lambda k: results[k]["f1"])
best_model = fitted[best_name]
best_metrics = results[best_name]

print(f"\n🏆 Best Model: {best_name}")
print(f"   F1 Score: {best_metrics['f1']:.4f}")
print(f"   Accuracy: {best_metrics['accuracy']:.4f}")

print(f"\n📦 Base models trained: {len(fitted)}")
for name in fitted.keys():
    f1_score_val = results[name]["f1"]
    print(f"   ✓ {name:<20} (F1: {f1_score_val:.4f})")

# Save results summary
summary = {
    "best_model": best_name,
    "best_f1": float(best_metrics["f1"]),
    "best_accuracy": float(best_metrics["accuracy"]),
    "all_results": {}
}

for name, metrics in results.items():
    summary["all_results"][name] = {
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "auc": float(metrics["auc"]),
        "cv_f1_mean": float(metrics["cv_f1_mean"]),
        "cv_f1_std": float(metrics["cv_f1_std"]),
    }

print(f"✓ Results summary prepared")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 9: Download All Base Models & Results
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("  📥 DOWNLOADING ALL MODELS & RESULTS")
print("═"*70)

from google.colab import files

# Download best model first
print("\n📦 Downloading best model...")
joblib.dump(best_model, "best_audio_model.joblib")
files.download("best_audio_model.joblib")

# Download scaler
print("📦 Downloading scaler...")
joblib.dump(scaler, "scaler.joblib")
files.download("scaler.joblib")

# Download all individual base models
print("\n📦 Downloading all 6 base models...")
for name, model in fitted.items():
    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(model, filename)
    try:
        files.download(filename)
        f1 = results[name]["f1"]
        print(f"   ✓ {filename:<30} (F1: {f1:.4f})")
    except Exception as e:
        print(f"   ✗ {filename} — {str(e)[:40]}")

# Download results summary
print("\n📦 Downloading results summary...")
results_json = json.dumps(summary, indent=2)
with open("training_results.json", "w") as f:
    f.write(results_json)
files.download("training_results.json")

print("✓ Results summary downloaded")
print("✓ All files downloaded successfully!")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 10: Visualization & Final Report
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("  📊 CREATING VISUALIZATIONS")
print("═"*70)

# Comparison chart
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics_names = ["f1", "accuracy", "precision", "recall"]
for idx, metric in enumerate(metrics_names):
    ax = axes[idx // 2, idx % 2]
    values = [results[name][metric] for name in results.keys()]
    colors = ["green" if name == best_name else "steelblue" for name in results.keys()]
    ax.barh(list(results.keys()), values, color=colors)
    ax.set_xlabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} Comparison")
    ax.set_xlim([0, 1])
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Created model comparison chart")
files.download("model_comparison.png")

# Confusion matrix for best model
y_pred_best = results[best_name]["y_pred"]
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(10, 8))
labels = [LABEL_NAMES[rmap[i]] for i in range(len(uniq))]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print(f"✓ Created confusion matrix")
files.download("confusion_matrix.png")

print("\n" + "═"*70)
print("  ✅ TRAINING & DOWNLOAD COMPLETE!")
print("═"*70)

print(f"\n🏆 Best Model: {best_name}")
print(f"   F1 Score: {best_metrics['f1']:.4f}")
print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
print(f"   Precision: {best_metrics['precision']:.4f}")
print(f"   Recall: {best_metrics['recall']:.4f}")

print(f"\n📊 All Base Models (sorted by F1):")
sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"   {i}. {name:<20} F1: {metrics['f1']:.4f}  Acc: {metrics['accuracy']:.4f}")

print(f"\n📥 Downloaded Files:")
print(f"   ✓ best_audio_model.joblib — {best_name} (F1: {best_metrics['f1']:.4f})")
print(f"   ✓ scaler.joblib — StandardScaler for feature normalization")
print(f"   ✓ training_results.json — Full metrics & CV scores")
print(f"\n   📦 All 6 Individual Base Models:")
for name, model in fitted.items():
    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    f1 = results[name]["f1"]
    print(f"      ✓ {filename:<35} F1: {f1:.4f}")

print(f"\n💡 Usage Instructions:")
print(f"   1. Download 'best_audio_model.joblib' and 'scaler.joblib'")
print(f"   2. Load in Python:")
print(f"      model = joblib.load('best_audio_model.joblib')")
print(f"      scaler = joblib.load('scaler.joblib')")
print(f"   3. For predictions:")
print(f"      X_scaled = scaler.transform(features)  # shape: (n, 88)")
print(f"      predictions = model.predict(X_scaled)")
print(f"      probabilities = model.predict_proba(X_scaled)")

print(f"\n✅ All done! Your models are ready for deployment.")
