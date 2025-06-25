import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Import our safe feature extraction functions
from features_safe import extract_safe_advanced_features, extract_basic_features

DATA_PATH = "data"


def load_data(use_advanced_features=True):
    """Load and extract features from all audio files"""
    X = []
    y = []

    print("🔍 Extracting features from audio files...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path '{DATA_PATH}' not found!")
        return None, None

    emotion_counts = {}
    failed_files = []

    # Choose feature extraction method
    extract_func = extract_safe_advanced_features if use_advanced_features else extract_basic_features
    feature_type = "safe_advanced" if use_advanced_features else "basic"

    print(f"Using {feature_type} feature extraction...")

    for emotion in os.listdir(DATA_PATH):
        emotion_folder = os.path.join(DATA_PATH, emotion)
        if not os.path.isdir(emotion_folder):
            continue

        emotion_counts[emotion] = 0

        for file in os.listdir(emotion_folder):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_folder, file)
                print(f"Processing: {file_path}")
                features = extract_func(file_path)
                if features is not None:
                    print(f"✅ Extracted {len(features)} features from {file}")
                    X.append(features)
                    y.append(emotion)
                    emotion_counts[emotion] += 1
                else:
                    failed_files.append(file_path)
                    print(f"❌ Failed to extract features from {file}")

    print(f"📊 Data distribution: {emotion_counts}")
    if failed_files:
        print(f"⚠️  Failed to process {len(failed_files)} files: {failed_files}")

    if len(X) == 0:
        print("❌ No features extracted from any files!")
        return None, None

    return np.array(X), np.array(y)


def train_model(use_advanced_features=True):
    """Train the emotion recognition model"""
    # Load data
    X, y = load_data(use_advanced_features)
    if X is None or len(X) == 0:
        print("❌ No data found! Please check your data folder structure.")
        return

    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"📝 Emotions: {np.unique(y)}")

    # Check if we have enough data for train/test split
    if len(X) < 4:
        print("⚠️  Very few samples. Training on all data without test split.")
        X_train, X_test = X, np.array([])
        y_train, y_test = y, np.array([])
    else:
        # Train-test split
        if len(X) < len(np.unique(y)) * 2:
            print("⚠️ Not enough data for stratified test split. Training on all data.")
            X_train, X_test, y_train, y_test = X, np.array([]), y, np.array([])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42, stratify=y
            )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    print("🎯 Training SVM model...")
    model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)

    # Evaluate model if we have test data
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"📈 Model accuracy: {accuracy:.2%}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print("📈 Model trained on all available data")

    # Save model and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'emotions': np.unique(y).tolist(),
        'feature_type': 'safe_advanced' if use_advanced_features else 'basic',
        'n_features': X.shape[1]
    }

    with open("voice_emotion_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"✅ Model saved as voice_emotion_model.pkl")
    print(f"   - Features: {X.shape[1]} ({model_data['feature_type']})")
    print(f"   - Emotions: {len(np.unique(y))}")

    return model_data


if __name__ == "__main__":
    print("🚀 Training with SAFE ADVANCED features...")
    train_model(use_advanced_features=True)