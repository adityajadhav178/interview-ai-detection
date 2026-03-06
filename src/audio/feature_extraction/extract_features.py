import numpy as np
import scipy.signal
import librosa
import parselmouth
from parselmouth.praat import call

def extract_audio_features(audio_array, sample_rate=16000):
    """
    Extracts 20 specific audio features grouped into 5 categories: 
    MFCC, Pitch, Energy, Speech Patterns, Voice Quality.
    
    Inputs:
    - audio_array: 1D numpy array of cleaned audio signal
    - sample_rate: sampling rate (default 16kHz)
    
    Returns:
    - Dictionary with 20 feature names and their corresponding values
    """
    features = {}
    
    if len(audio_array) == 0:
        return None
        
    # --- 1. MFCC (6 features) ---
    # Compute 13 MFCCs
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
    # Compute Delta and Delta-Delta
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # "mean of 13 coefficients collapsed to 1 value" means taking the mean of all MFCC values
    features["mfcc_mean"] = np.mean(mfccs)
    features["mfcc_std"] = np.std(mfccs)
    features["delta_mfcc_mean"] = np.mean(delta_mfccs)
    features["delta_mfcc_std"] = np.std(delta_mfccs)
    features["delta_delta_mfcc_mean"] = np.mean(delta2_mfccs)
    features["delta_delta_mfcc_std"] = np.std(delta2_mfccs)
    
    # --- 2. Pitch (4 features) ---
    # Using Parselmouth (Praat) for accurate clinical-grade pitch (F0) extraction.
    # This is more reliable than librosa.piptrack for speech signals.
    try:
        snd_pitch = parselmouth.Sound(audio_array, sample_rate)
        pitch_obj = snd_pitch.to_pitch()
        # Get frequency values for all frames; unvoiced frames = 0.0
        pitch_values_raw = pitch_obj.selected_array['frequency']
        # Filter out zero (unvoiced) values
        pitch_values = pitch_values_raw[pitch_values_raw > 0]

        if len(pitch_values) > 0:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"]  = float(np.std(pitch_values))
            features["pitch_min"]  = float(np.min(pitch_values))
            features["pitch_max"]  = float(np.max(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"]  = 0.0
            features["pitch_min"]  = 0.0
            features["pitch_max"]  = 0.0
    except Exception as e:
        print(f"Warning: Failed to extract Parselmouth pitch: {e}")
        features["pitch_mean"] = 0.0
        features["pitch_std"]  = 0.0
        features["pitch_min"]  = 0.0
        features["pitch_max"]  = 0.0
        
    # --- 3. Energy (4 features) ---
    # Using RMS energy per frame
    rms_frames = librosa.feature.rms(y=audio_array)[0]
    features["energy_mean"] = np.mean(rms_frames)
    features["energy_std"] = np.std(rms_frames)
    features["energy_min"] = np.min(rms_frames)
    features["rms_energy"] = np.sqrt(np.mean(audio_array**2)) # Overall RMS energy
    
    # --- 4. Speech Patterns (4 features) ---
    # Speech Rate: Number of voiced segments / total duration
    # Since audio is already "cleaned" (VAD applied), we approximate syllables via envelope peaks
    # Or zero crossing rate
    duration = len(audio_array) / sample_rate
    
    # Zero Crossing Rate Mean
    zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
    features["zero_crossing_rate_mean"] = np.mean(zcr)
    
    # Simple silence detection on the provided array to calculate pause statistics
    # Thresholding at 10% of max amplitude to find "pauses"
    threshold = 0.1 * np.max(np.abs(audio_array))
    is_silent = np.abs(audio_array) < threshold
    
    # Find continuous silent segments
    silent_segments = np.split(is_silent, np.where(np.diff(is_silent))[0] + 1)
    pauses = [len(seg)/sample_rate for seg in silent_segments if seg[0] == True and len(seg)/sample_rate > 0.05] # Pauses > 50ms
    
    features["pause_frequency"] = len(pauses)
    features["pause_duration_mean"] = np.mean(pauses) if pauses else 0.0
    
    # Speech Rate (pseudo-syllable rate): peaks in energy
    # Find peaks in RMS energy
    peaks, _ = scipy.signal.find_peaks(rms_frames, distance=10) # Roughly 10 frames apart
    features["speech_rate"] = len(peaks) / duration if duration > 0 else 0.0

    # --- 5. Voice Quality (2 features) ---
    # Using Parselmouth (Praat) for Jitter and Shimmer
    # Jitter: frequency variations, Shimmer: amplitude variations
    try:
        snd = parselmouth.Sound(audio_array, sample_rate)
        # To calculate jitter and shimmer, we need point process (pitch tracking)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        pointProcess = call(snd, "To PointProcess (periodic, cc)", pitch)
        
        # local jitter
        jitter = call(pointProcess, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        # local shimmer
        shimmer = call([snd, pointProcess], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        
        features["jitter"] = jitter if not np.isnan(jitter) else 0.0
        features["shimmer"] = shimmer if not np.isnan(shimmer) else 0.0
    except Exception as e:
        print(f"Warning: Failed to extract Parselmouth features: {e}")
        features["jitter"] = 0.0
        features["shimmer"] = 0.0
        
    return features

# Example usage:
if __name__ == "__main__":
    import sys
    # Example to test the feature extractor with dummy audio
    dummy_audio = np.random.randn(16000 * 3) # 3 seconds of dummy noise
    feats = extract_audio_features(dummy_audio, 16000)
    print("Extracted Features:")
    for k, v in feats.items():
        print(f"{k}: {v}")
