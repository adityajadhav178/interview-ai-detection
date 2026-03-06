import os
import joblib
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Import pipeline components
from src.audio.preprocessing.audio_cleaner import preprocess_audio
from src.audio.feature_extraction.extract_features import extract_audio_features
from src.audio.feature_extraction.build_dataset import aggregate_features

class AudioMentalHealthPredictor:
    def __init__(self, model_dir="models/audio"):
        """
        Initializes the predictor by loading the saved scaler and best model.
        """
        self.model_dir = Path(model_dir)
        scaler_path = self.model_dir / "standard_scaler.joblib"
        model_path = self.model_dir / "best_audio_model.joblib"
        
        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Missing modeling artifacts in {model_dir}. Please run training step first.")
            
        print("Loading standard scaler...")
        self.scaler = joblib.load(scaler_path)
        
        print("Loading best audio model...")
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.label_mapping = model_data.get("label_mapping", None) # dictmapping e.g., {0: 1, 1: 0, 2: 2} if classes were mapped

    def extract_patient_features(self, patient_folder):
        """
        Runs the full preprocessing and feature extraction pipeline on checking 
        up to 5 wav files in the patient_folder.
        """
        folder_path = Path(patient_folder)
        wav_files = list(folder_path.glob("*.wav"))
        
        if len(wav_files) == 0:
            raise ValueError(f"No .wav files found in {patient_folder}")
            
        print(f"Found {len(wav_files)} audio files. Processing...")
        
        patient_features = []
        for wav_file in wav_files[:5]: # Standard 5 answers expected
            try:
                # 1. Clean audio (load, VAD, noise reduction, RMS normalize, Pre-emphasis)
                clean_audio, sr = preprocess_audio(str(wav_file))
                
                # 2. Extract 20 features
                if clean_audio is not None and len(clean_audio) > 0:
                    feats = extract_audio_features(clean_audio, sr)
                    if feats:
                        patient_features.append(feats)
                        print(f"  - Extracted features from {wav_file.name}")
                else:
                    print(f"  - Warning: Failed to clean or empty audio {wav_file.name}")
            except Exception as e:
                print(f"  - Error processing {wav_file.name}: {e}")
                
        if not patient_features:
            raise ValueError("Failed to extract features from any audio file.")
            
        # 3. Aggregate across answers (100 features)
        print("Aggregating features...")
        aggregated_feats = aggregate_features(patient_features)
        
        return aggregated_feats

    def predict(self, patient_folder):
        """
        Runs the full pipeline to predict mental health condition.
        """
        try:
            # 1. Extract & Aggregate Features
            features_dict = self.extract_patient_features(patient_folder)
            
            # 2. Convert to DataFrame (ensure correct 100 feature columns layout)
            df = pd.DataFrame([features_dict])
            
            # Note: At inference time, ensure the columns match the exact ones seen during training.
            # In production, it's safer to save column names during training and reorder df here.
            expected_features = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else df.columns
            df = df.reindex(columns=expected_features, fill_value=0.0)
            
            # 3. Scale Features
            X_scaled = self.scaler.transform(df)
            
            # 4. Predict Label and Probabilities
            prediction = self.model.predict(X_scaled)[0]
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_scaled)[0]
            else:
                probs = None
                
            # Map back to original label if a mapping exists
            original_label = prediction
            if self.label_mapping and prediction in self.label_mapping:
                original_label = self.label_mapping[prediction]
                
            # Formatting probability output
            prob_dict = {}
            if probs is not None:
                if self.label_mapping:
                    for pred_idx, original_idx in self.label_mapping.items():
                        prob_dict[original_idx] = probs[pred_idx]
                else:
                    for idx, p in enumerate(probs):
                        prob_dict[idx] = p
                        
            return {
                "predicted_label": original_label,
                "probabilities": prob_dict
            }
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mental health condition from audio files")
    parser.add_argument("--patient_folder", type=str, required=True, help="Folder containing up to 5 .wav files")
    parser.add_argument("--model_dir", type=str, default="models/audio", help="Directory containing saved model artifacts")
    
    args = parser.parse_args()
    
    try:
        predictor = AudioMentalHealthPredictor(args.model_dir)
        print(f"\n--- Running Prediction for {args.patient_folder} ---")
        result = predictor.predict(args.patient_folder)
        
        if result:
            print(f"\n✅ Final Prediction: Class {result['predicted_label']}")
            if result['probabilities']:
                print("\nConfidence Scores:")
                for cls, prob in result['probabilities'].items():
                    print(f"  - Class {cls}: {prob*100:.2f}%")
    except Exception as e:
        print(f"\nFailed to run predictor: {e}")
