"""
Safe feature extraction module that works with any librosa version
"""
import librosa
import numpy as np


def extract_basic_features(file_path):
    """Extract basic MFCC features (guaranteed to work)"""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_safe_advanced_features(file_path):
    """Extract enhanced features that work with older librosa versions"""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)

        # 1. MFCC features (40 coefficients) - Always works
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

        # 2. Additional statistical features from the audio signal itself
        # These don't depend on specific librosa feature functions

        # Energy-based features
        energy = np.sum(audio ** 2) / len(audio)  # RMS energy

        # Zero crossing rate (manual calculation)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

        # Spectral features from raw FFT
        fft = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(fft), 1 / sr)

        # Keep only positive frequencies
        fft = fft[:len(fft) // 2]
        freqs = freqs[:len(freqs) // 2]

        # Avoid division by zero
        if np.sum(fft) == 0:
            spectral_centroid = 0
            spectral_rolloff = 0
            spectral_bandwidth = 0
        else:
            # Spectral centroid (manually calculated)
            spectral_centroid = np.sum(freqs * fft) / np.sum(fft)

            # Spectral rolloff (manually calculated)
            cumsum_fft = np.cumsum(fft)
            rolloff_point = 0.85 * cumsum_fft[-1]
            rolloff_idx = np.where(cumsum_fft >= rolloff_point)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0

            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft) / np.sum(fft))

        # Simplified tempo estimation without frames
        # Use autocorrelation method
        audio_autocorr = np.correlate(audio, audio, mode='full')
        audio_autocorr = audio_autocorr[len(audio_autocorr)//2:]

        # Find peaks in autocorrelation to estimate tempo
        if len(audio_autocorr) > 1:
            tempo_estimate = np.argmax(audio_autocorr[1:]) + 1  # Avoid zero lag
            tempo_estimate = sr / tempo_estimate if tempo_estimate > 0 else 120  # Convert to BPM-like measure
        else:
            tempo_estimate = 120

        # Statistical features of the audio signal
        audio_mean = np.mean(audio)
        audio_std = np.std(audio)

        if audio_std > 0:
            audio_skew = np.mean(((audio - audio_mean) / audio_std) ** 3)
            audio_kurtosis = np.mean(((audio - audio_mean) / audio_std) ** 4)
        else:
            audio_skew = 0
            audio_kurtosis = 0

        # Ensure all features are scalars (not arrays)
        features_list = [
            mfccs,  # 40 features (array)
            np.array([energy]),  # 1 feature
            np.array([zcr]),  # 1 feature
            np.array([spectral_centroid]),  # 1 feature
            np.array([spectral_rolloff]),  # 1 feature
            np.array([spectral_bandwidth]),  # 1 feature
            np.array([tempo_estimate]),  # 1 feature
            np.array([audio_mean]),  # 1 feature
            np.array([audio_std]),  # 1 feature
            np.array([audio_skew]),  # 1 feature
            np.array([audio_kurtosis])  # 1 feature
        ]

        # Combine all features - flatten everything to 1D
        combined_features = np.concatenate([f.flatten() for f in features_list])

        return combined_features  # Total: 50 features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_features_from_bytes(audio_bytes):
    """Extract features from audio bytes (for Streamlit uploads)"""
    try:
        import io
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        # Use the same safe feature extraction as above
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

        # Manual feature calculations (same as above)
        energy = np.sum(audio ** 2) / len(audio)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

        fft = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(fft), 1 / sr)
        fft = fft[:len(fft) // 2]
        freqs = freqs[:len(freqs) // 2]

        if np.sum(fft) == 0:
            spectral_centroid = 0
            spectral_rolloff = 0
            spectral_bandwidth = 0
        else:
            spectral_centroid = np.sum(freqs * fft) / np.sum(fft)

            cumsum_fft = np.cumsum(fft)
            rolloff_point = 0.85 * cumsum_fft[-1]
            rolloff_idx = np.where(cumsum_fft >= rolloff_point)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0

            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft) / np.sum(fft))

        # Simplified tempo estimation
        audio_autocorr = np.correlate(audio, audio, mode='full')
        audio_autocorr = audio_autocorr[len(audio_autocorr)//2:]

        if len(audio_autocorr) > 1:
            tempo_estimate = np.argmax(audio_autocorr[1:]) + 1
            tempo_estimate = sr / tempo_estimate if tempo_estimate > 0 else 120
        else:
            tempo_estimate = 120

        audio_mean = np.mean(audio)
        audio_std = np.std(audio)

        if audio_std > 0:
            audio_skew = np.mean(((audio - audio_mean) / audio_std) ** 3)
            audio_kurtosis = np.mean(((audio - audio_mean) / audio_std) ** 4)
        else:
            audio_skew = 0
            audio_kurtosis = 0

        # Ensure all features are properly shaped
        features_list = [
            mfccs,  # 40 features
            np.array([energy]),
            np.array([zcr]),
            np.array([spectral_centroid]),
            np.array([spectral_rolloff]),
            np.array([spectral_bandwidth]),
            np.array([tempo_estimate]),
            np.array([audio_mean]),
            np.array([audio_std]),
            np.array([audio_skew]),
            np.array([audio_kurtosis])
        ]

        # Combine all features - flatten everything to 1D
        combined_features = np.concatenate([f.flatten() for f in features_list])

        return combined_features

    except Exception as e:
        print(f"Error processing audio bytes: {e}")
        return None