import numpy as np
import scipy.signal
import librosa
import parselmouth


def extract_audio_features(audio_array, sample_rate=16000):
    """
    Extracts 18 audio features grouped into 4 categories:
    MFCC (6), Pitch (4), Energy (4), Speech Patterns (4)

    Inputs:
    - audio_array : 1D numpy float32/float64 array of cleaned audio signal
    - sample_rate : sampling rate in Hz (default 16 kHz)

    Returns:
    - Dictionary with 18 feature names and float values
    - Returns None if audio_array is empty
    """

    if audio_array is None or len(audio_array) == 0:
        return None

    audio_array = audio_array.astype(np.float64)
    features = {}

    # ---------------------------------------------------------------
    # 1. MFCC FEATURES (6)
    # ---------------------------------------------------------------
    mfccs = librosa.feature.mfcc(
        y=audio_array.astype(np.float32),
        sr=sample_rate,
        n_mfcc=13
    )

    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    features["mfcc_mean"] = float(np.mean(mfccs))
    features["mfcc_std"] = float(np.std(mfccs))

    features["delta_mfcc_mean"] = float(np.mean(delta_mfccs))
    features["delta_mfcc_std"] = float(np.std(delta_mfccs))

    features["delta_delta_mfcc_mean"] = float(np.mean(delta2_mfccs))
    features["delta_delta_mfcc_std"] = float(np.std(delta2_mfccs))

    # ---------------------------------------------------------------
    # 2. PITCH FEATURES (4)
    # ---------------------------------------------------------------
    try:
        snd = parselmouth.Sound(audio_array, sampling_frequency=sample_rate)
        pitch = snd.to_pitch()

        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) > 0:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
            features["pitch_min"] = float(np.min(pitch_values))
            features["pitch_max"] = float(np.max(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_min"] = 0.0
            features["pitch_max"] = 0.0

    except Exception as e:
        print(f"Warning: Failed to extract pitch: {e}")
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0

    # ---------------------------------------------------------------
    # 3. ENERGY FEATURES (4)
    # ---------------------------------------------------------------
    rms_frames = librosa.feature.rms(
        y=audio_array.astype(np.float32)
    )[0]

    features["energy_mean"] = float(np.mean(rms_frames))
    features["energy_std"] = float(np.std(rms_frames))
    features["energy_min"] = float(np.min(rms_frames))

    features["rms_energy"] = float(
        np.sqrt(np.mean(audio_array ** 2))
    )

    # ---------------------------------------------------------------
    # 4. SPEECH PATTERN FEATURES (4)
    # ---------------------------------------------------------------
    duration = len(audio_array) / sample_rate

    zcr = librosa.feature.zero_crossing_rate(
        audio_array.astype(np.float32)
    )[0]

    features["zero_crossing_rate_mean"] = float(np.mean(zcr))

    # Pause detection
    peak_amp = np.max(np.abs(audio_array))
    threshold = 0.1 * peak_amp if peak_amp > 0 else 1e-6

    is_silent = np.abs(audio_array) < threshold

    boundaries = np.where(np.diff(is_silent.astype(int)))[0] + 1
    segments = np.split(is_silent, boundaries)

    pauses = [
        len(seg) / sample_rate
        for seg in segments
        if len(seg) > 0 and bool(seg[0]) and (len(seg) / sample_rate) > 0.05
    ]

    features["pause_frequency"] = len(pauses)

    if pauses:
        features["pause_duration_mean"] = float(np.mean(pauses))
    else:
        features["pause_duration_mean"] = 0.0

    # Speech rate estimation
    peaks, _ = scipy.signal.find_peaks(rms_frames, distance=10)

    if duration > 0:
        features["speech_rate"] = float(len(peaks) / duration)
    else:
        features["speech_rate"] = 0.0

    return features


# ------------------------------------------------------------------
# Smoke Test
# Run this file directly to verify the extractor works
# ------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("Smoke Test — Random 3 Second Audio")
    print("=" * 50)

    dummy_audio = np.random.randn(16000 * 3)

    feats = extract_audio_features(dummy_audio, 16000)

    if feats:
        print(f"\nTotal features extracted: {len(feats)}\n")

        for k, v in feats.items():
            print(f"{k:<35} : {v:.6f}")

    else:
        print("Feature extraction failed.")