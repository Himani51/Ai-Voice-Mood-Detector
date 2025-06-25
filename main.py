import joblib
import numpy as np

# Import our feature extraction functions
from features import extract_advanced_features, extract_basic_features


def test_feature_extraction(file_path, use_advanced=True):
    """Test feature extraction with basic or advanced features"""
    extract_func = extract_advanced_features if use_advanced else extract_basic_features
    feature_type = "advanced" if use_advanced else "basic"

    print(f"Testing {feature_type} feature extraction...")
    features = extract_func(file_path)

    if features is not None:
        print(f"✅ Extracted {len(features)} {feature_type} features")
        print(f"Feature shape: {features.shape}")
        print(f"Sample features: {features[:5]}")  # Show first 5 features
        return features
    else:
        print("❌ Failed to extract features")
        return None


if __name__ == "__main__":
    audio_file = "C:/Users/vagha/Music/Sample-audio.wav"

    # Test both feature extraction methods
    print("=" * 50)
    basic_features = test_feature_extraction(audio_file, use_advanced=False)

    print("\n" + "=" * 50)
    advanced_features = test_feature_extraction(audio_file, use_advanced=True)

    # Test with trained model
    print("\n" + "=" * 50)
    print("Testing with trained model...")

    try:
        model_data = joblib.load("voice_emotion_model.pkl")
        model = model_data['model']
        scaler = model_data['scaler']
        feature_type = model_data.get('feature_type', 'basic')

        print(f"Model expects {feature_type} features")

        # Use appropriate features based on model
        if feature_type == 'advanced' and advanced_features is not None:
            features = advanced_features
        elif feature_type == 'basic' and basic_features is not None:
            features = basic_features
        else:
            print("❌ Feature type mismatch or extraction failed")
            exit()

        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled).max()

        print(f"🎯 Detected mood: {prediction} ({confidence:.2%} confidence)")

    except FileNotFoundError:
        print("❌ Model not found. Run train_model.py first to train the model.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")