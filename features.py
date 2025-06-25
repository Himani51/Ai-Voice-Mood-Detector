"""
Feature extraction module for voice emotion detection
"""
import librosa
import numpy as np

def extract_basic_features(file_path):
    """Extract basic MFCC features (your original method)"""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_advanced_features(file_path):
    """Extract comprehensive audio features for better emotion detection"""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)

        # 1. MFCC features (40 coefficients)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

        # 2. Try different feature extraction methods based on librosa version
        try:
            # Try new librosa syntax
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        except AttributeError:
            # Fallback for older librosa versions
            stft = librosa.stft(audio)
            chroma = np.mean(librosa.feature.chroma(C=np.abs(stft)).T, axis=0)

        # 3. Spectral contrast
        try:
            contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        except:
            # If spectral_contrast fails, create dummy features
            contrast = np.zeros(7)

        # 4. Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # 5. Spectral rolloff
        try:
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        except:
            rolloff = 0.0

        # 6. Spectral centroid
        try:
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        except:
            centroid = 0.0

        # 7. Tempo - simplified version
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        except:
            tempo = 120.0  # Default tempo

        # Combine all features
        combined_features = np.hstack([
            mfccs,           # 40 features
            chroma,          # 12 features
            contrast,        # 7 features
            [zcr],           # 1 feature
            [rolloff],       # 1 feature
            [centroid],      # 1 feature
            [tempo]          # 1 feature
        ])

        return combined_features  # Total: 63 features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_from_bytes(audio_bytes):
    """Extract features from audio bytes (for Streamlit uploads)"""
    try:
        import io
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        # Use the same feature extraction as above with error handling
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

        # Chroma with fallback
        try:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        except AttributeError:
            stft = librosa.stft(audio)
            chroma = np.mean(librosa.feature.chroma(C=np.abs(stft)).T, axis=0)

        # Other features with error handling
        try:
            contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        except:
            contrast = np.zeros(7)

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        try:
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        except:
            rolloff = 0.0

        try:
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        except:
            centroid = 0.0

        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        except:
            tempo = 120.0

        combined_features = np.hstack([
            mfccs, chroma, contrast, [zcr], [rolloff], [centroid], [tempo]
        ])

        return combined_features

    except Exception as e:
        print(f"Error processing audio bytes: {e}")
        return None