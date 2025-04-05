from fastapi import HTTPException
import numpy as np
import joblib
import tensorflow as tf
import os
import pandas as pd

# Suppress absl warnings (e.g., compile_metrics warning)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING messages



model_age = joblib.load("age_xgb_model.pkl")
scaler_age = joblib.load("age_scaler.pkl")
label_encoder_age = joblib.load("age_label_encoder.pkl")


def load_model_and_scaler(model1_path='voice_classification_model.h5', scaler_path='voice_scaler.pkl', max_uncertainty_path = 'max_uncertainty.pkl'):
    """
    Load the saved model and scaler
    """
    # Load model
    loaded_model1 = tf.keras.models.load_model(model1_path)

    # Load scaler
    loaded_scaler = joblib.load(scaler_path)
    max_uncertainty = joblib.load(max_uncertainty_path)

    return loaded_model1, loaded_scaler, max_uncertainty

def predict_with_uncertainty(model1, X, num_samples=100):
    """
    Make predictions with uncertainty estimation using Monte Carlo Dropout
    """
    preds = []
    for _ in range(num_samples):
        preds.append(model1(X, training=False))  # Keep dropout enabled for MC sampling

    preds = np.array([p.numpy() for p in preds])
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    # Normalize uncertainty to percentage
    max_uncertainty = np.max(pred_std) if np.max(pred_std) > 0 else 1.0
    uncertainty_percent = (pred_std / max_uncertainty) * 100

    return pred_mean, pred_std ,uncertainty_percent

def predict_gender_from_voice_features(voice_features, model1=None, scaler=None):
    if model1 is None or scaler is None:
        model1, scaler , max_uncertainty = load_model_and_scaler()

    # Process features
    feature_names = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
                     'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
                     'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
    features_list = [voice_features.get(name, 0) for name in feature_names]
    X = np.array(features_list).reshape(1, -1)

    # Scale and convert
    X_scaled = scaler.transform(X)
    X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
    # Predict with MC Dropout
    pred_mean, pred_std ,normalized_uncertainty = predict_with_uncertainty(model1, X_tensor)
    # normalized_uncertainty = np.minimum((pred_std / max_uncertainty) * 100, 100)
    gender = "male" if pred_mean[0][0] > 0.5 else "female"

    return {
        "probability": float(pred_mean[0][0]),
        "gender": gender,
        "uncertainty_raw": float(pred_std[0][0]),
        "uncertainty_percent": float(normalized_uncertainty[0][0]),
        "confidence": 100 - float(normalized_uncertainty[0][0])
        
    }

def predict_age(features):
    try:
        # Extract features in the same order as training

        feature_names = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
                         'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
                         'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']

        X = pd.DataFrame([[features.get(name, 0) for name in feature_names]], columns=feature_names)

        # Scale features
        X_scaled = scaler_age.transform(X)

        # Predict
        pred = model_age.predict(X_scaled)
        age_label = label_encoder_age.inverse_transform(pred)[0]

        return {"predicted_age_group": age_label}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))