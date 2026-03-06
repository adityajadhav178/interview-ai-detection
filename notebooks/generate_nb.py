import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + ("\n" if i < len(text.split("\n")) - 1 else "") for i, line in enumerate(text.split("\n"))]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + ("\n" if i < len(text.split("\n")) - 1 else "") for i, line in enumerate(text.split("\n"))]
    })

add_md("""# 📘 Audio Mental Health Detection Pipeline\nClassical Machine Learning Pipeline for Mental Health Prediction from Speech\n\n**Steps**\n1. Install dependencies\n2. Imports\n3. Audio preprocessing\n4. Audio visualization\n5. Feature extraction\n6. Feature visualization\n7. Dataset creation\n8. SMOTE balancing\n9. Model training\n10. Inference""")
add_md("""---\n# 1️⃣ Install Dependencies""")
add_code("""# %pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm librosa noisereduce tqdm praat-parselmouth ipython webrtcvad-wheels imbalanced-learn""")
add_md("""---\n# 2️⃣ Imports""")
add_code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display
import IPython.display as ipd

import noisereduce as nr
import webrtcvad
import scipy.signal

import parselmouth
from parselmouth.praat import call

from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import xgboost as xgb
import lightgbm as lgb

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline""")

add_md("""---\n# 3️⃣ Audio Preprocessing\n\nThis step performs:\n* Noise reduction\n* Voice activity detection\n* RMS normalization\n* Pre-emphasis filtering""")
add_code("""def run_vad_on_audio(audio_array, sample_rate, aggressiveness=3):

    frame_duration_ms = 30
    frame_size = int(sample_rate * (frame_duration_ms / 1000))

    vad = webrtcvad.Vad(aggressiveness)

    audio_int16 = (audio_array * 32767).astype(np.int16)

    padding = frame_size - (len(audio_int16) % frame_size)

    if padding < frame_size:
        audio_int16 = np.pad(audio_int16, (0, padding), mode='constant')

    voiced_frames = []

    for i in range(0, len(audio_int16), frame_size):

        frame = audio_int16[i:i+frame_size]

        if len(frame) == frame_size and vad.is_speech(frame.tobytes(), sample_rate):
            voiced_frames.append(frame)

    if len(voiced_frames) == 0:
        return audio_array

    voiced_audio = np.concatenate(voiced_frames)

    return voiced_audio.astype(np.float32) / 32767""")

add_md("""### RMS Normalization""")
add_code("""def normalize_rms(audio, target_rms=0.1):

    rms = np.sqrt(np.mean(audio**2))

    if rms > 0:
        audio = audio * (target_rms / rms)

    return audio""")

add_md("""### Pre-emphasis Filter""")
add_code("""def apply_preemphasis(audio, coeff=0.97):

    return np.append(audio[0], audio[1:] - coeff * audio[:-1])""")

add_md("""### Main Preprocessing Function""")
add_code("""def preprocess_audio(file_path):

    sample_rate = 16000

    y, sr = librosa.load(file_path, sr=sample_rate)

    if len(y) == 0:
        return None, None, None

    y_denoised = nr.reduce_noise(y=y, sr=sr)

    y_vad = run_vad_on_audio(y_denoised, sr)

    y_norm = normalize_rms(y_vad)

    y_clean = apply_preemphasis(y_norm)

    return y_clean, y, sr""")

add_md("""---\n# 4️⃣ Audio Visualization\n### Waveform""")
add_code("""def visualize_audio(file):

    y_clean, y_orig, sr = preprocess_audio(file)

    plt.figure(figsize=(14,6))

    plt.subplot(2,1,1)
    librosa.display.waveshow(y_orig, sr=sr)
    plt.title("Original Audio")

    plt.subplot(2,1,2)
    librosa.display.waveshow(y_clean, sr=sr)
    plt.title("Cleaned Audio")

    plt.tight_layout()
    plt.show()

    print("Original Audio")
    ipd.display(ipd.Audio(y_orig, rate=sr))

    print("Cleaned Audio")
    ipd.display(ipd.Audio(y_clean, rate=sr))""")

add_md("""### Spectrogram""")
add_code("""def visualize_spectrogram(file):

    y_clean, y_orig, sr = preprocess_audio(file)

    plt.figure(figsize=(14,5))

    S = librosa.stft(y_clean)
    S_db = librosa.amplitude_to_db(abs(S))

    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')

    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()""")

add_md("""---\n# 5️⃣ Feature Extraction\n\nExtracts **speech diagnostic features**.""")
add_code("""def extract_audio_features(audio, sr):

    features = {}

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    features["mfcc_mean"] = np.mean(mfcc)
    features["mfcc_std"] = np.std(mfcc)

    delta = librosa.feature.delta(mfcc)
    features["delta_mean"] = np.mean(delta)
    features["delta_std"] = np.std(delta)

    sound = parselmouth.Sound(audio, sr)
    pitch = sound.to_pitch()

    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values>0]

    if len(pitch_values)>0:

        features["pitch_mean"] = np.mean(pitch_values)
        features["pitch_std"] = np.std(pitch_values)

    else:

        features["pitch_mean"] = 0
        features["pitch_std"] = 0

    rms = librosa.feature.rms(y=audio)[0]

    features["energy_mean"] = np.mean(rms)
    features["energy_std"] = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    features["zcr_mean"] = np.mean(zcr)

    return features""")

add_md("""---\n# 6️⃣ Feature Visualization""")
add_code("""def visualize_features(file):

    y_clean, _, sr = preprocess_audio(file)

    feats = extract_audio_features(y_clean, sr)

    df = pd.DataFrame([feats]).T
    df.columns=["value"]

    plt.figure(figsize=(8,5))

    sns.barplot(x=df["value"], y=df.index)

    plt.title("Extracted Audio Features")

    plt.show()""")

add_md("""---\n# 7️⃣ Dataset Creation""")
add_code("""def create_dataset(folder, label):

    rows=[]

    for file in tqdm(os.listdir(folder)):

        if file.endswith(".wav"):

            path=os.path.join(folder,file)

            y_clean,_,sr=preprocess_audio(path)

            feats=extract_audio_features(y_clean,sr)

            feats["label"]=label

            rows.append(feats)

    return pd.DataFrame(rows)""")

add_md("""### Example""")
add_code("""normal_df=create_dataset("../data/raw/audio/normal",0)
depressed_df=create_dataset("../data/raw/audio/depression",1)

df=pd.concat([normal_df,depressed_df])

df.head()""")

add_md("""---\n# 8️⃣ SMOTE Balancing""")
add_code("""X=df.drop("label",axis=1)
y=df["label"]

smote=SMOTE(random_state=42)

X_res,y_res=smote.fit_resample(X,y)

print("Original:",y.value_counts())
print("Balanced:",pd.Series(y_res).value_counts())""")

add_md("""---\n# 9️⃣ Model Training""")
add_code("""X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.2,random_state=42)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

models={
"RandomForest":RandomForestClassifier(),
"XGBoost":xgb.XGBClassifier(eval_metric="logloss"),
"LightGBM":lgb.LGBMClassifier()
}

for name,model in models.items():

    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    print(name)

    print("Accuracy:",accuracy_score(y_test,preds))

    print("F1:",f1_score(y_test,preds))""")

add_md("""---\n# 🔟 Inference on New Audio""")
add_code("""def predict_audio(file,model,scaler):

    y_clean,_,sr=preprocess_audio(file)

    feats=extract_audio_features(y_clean,sr)

    df=pd.DataFrame([feats])

    df=scaler.transform(df)

    pred=model.predict(df)

    print("Prediction:",pred)""")

target_path = r"c:\DOCUMENTS\Harsh\interview-ai-detection\notebooks\audio_experiments.ipynb"

with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully generated new notebook layout at {target_path}")
