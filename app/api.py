import os
import torch
import joblib
import pandas as pd
import numpy as np
import librosa
import xgboost as xgb
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "Models")

# Global model variables
feature_names = None
sleep_scaler = None
sleep_model = None
tokenizer = None
text_model = None
voice_model = None

def load_models():
    """Load all ML models into memory."""
    global feature_names, sleep_scaler, sleep_model, tokenizer, text_model, voice_model
    print("Initializing SleepSense AI Triple-Fusion Engine...")
    
    # Load Physical Assets
    try:
        feature_names = joblib.load(os.path.join(MODELS_PATH, 'feature_list_v3.pkl'))
        sleep_scaler = joblib.load(os.path.join(MODELS_PATH, 'sleep_scaler_v3.pkl'))
        sleep_model = xgb.XGBClassifier()
        sleep_model.load_model(os.path.join(MODELS_PATH, 'sleep_model_v3.json'))
        print("Physical Model Loaded (XGBoost)")
    except Exception as e:
        print(f"Error loading Physical Model: {e}")

    # Load Mental Assets
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_PATH, "mental_bert_model"))
        text_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODELS_PATH, "mental_bert_model"))
        print("Mental Model Loaded (MentalBERT)")
    except Exception as e:
        print(f"Error loading Mental Model: {e}")

    # Load Vocal Assets
    try:
        voice_model = load_model(os.path.join(MODELS_PATH, 'voice_model.h5'))
        print("Vocal Model Loaded (1D-CNN)")
    except Exception as e:
        print(f"Error loading Vocal Model: {e}")

def get_voice_features(file_path_or_bytes):
    """Extracts MFCC features for the CNN model."""
    # librosa.load can take a file-like object or a path
    X, sr = librosa.load(file_path_or_bytes, res_type='kaiser_fast', duration=2.5, sr=22050, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(np.expand_dims(mfccs, axis=0), axis=2)

def hybrid_fusion_logic(p_risk, t_risk, v_risk):
    """Combines model outputs with clinical red-flag overrides."""
    if t_risk > 0.92 or p_risk > 0.95:
        return max(p_risk, t_risk, v_risk), "CRITICAL RISK", "URGENT: Extreme distress detected. Immediate support recommended."
    
    total_score = (p_risk * 0.40) + (t_risk * 0.30) + (v_risk * 0.30)
    
    if total_score > 0.70:
        status, advice = "CRITICAL RISK", "High cumulative load. Urgent rest and consultation advised."
    elif total_score > 0.40:
        status = "MODERATE RISK"
        if t_risk > p_risk: advice = "Mental fatigue is dominant. Consider cognitive breaks."
        else: advice = "Physical burnout detected. Prioritize sleep hygiene."
    else:
        status, advice = "STABLE", "All indicators are within normal safety ranges."
        
    return total_score, status, advice

def generate_report_logic(user_input: dict, text_msg: str, audio_file_path: str):
    """
    Processes user data, runs inferences, and generates the final report data.
    """
    # 1. Extract Demographics & Biometrics
    age = user_input.get('age', 30)
    gen = user_input.get('gender', 1)
    occ = user_input.get('occupation', 0)
    work = user_input.get('work_hours', 8.0)
    dur = user_input.get('sleep_duration', 7.0)
    lat = user_input.get('sleep_latency', 20)
    wake = user_input.get('wake_count', 1)
    bed_m = user_input.get('bedtime_num', 1380)
    wak_m = user_input.get('waketime_num', 420)
    stress = user_input.get('stress_level_num', 1)

    # Smartwatch data (Auto-fill if missing or not provided)
    deep = user_input.get('deep_sleep_percent')
    if deep is None: deep = 20.0
    rem = user_input.get('rem_sleep_percent')
    if rem is None: rem = 22.0
    
    # Auto-calculate Efficiency if missing
    eff = user_input.get('sleep_efficiency')
    if eff is None:
        in_bed_mins = (wak_m - bed_m) % 1440
        if in_bed_mins == 0: in_bed_mins = 480
        calculated_eff = (dur / (in_bed_mins / 60)) * 100
        eff = min(calculated_eff, 98.0)

    # 2. Advanced Feature Engineering
    deficit = 8.0 - dur
    intensity = work / (dur + 0.5)
    restless = wake * lat
    drift = min(abs(bed_m - 1380), 1440 - abs(bed_m - 1380))
    
    p_risk = 0.0
    t_risk = 0.0
    v_risk = 0.0

    # 3. Physical Prediction
    if sleep_model is not None and sleep_scaler is not None and feature_names is not None:
        p_features = [age, gen, occ, work, dur, lat, eff, wake, bed_m, wak_m, 
                    deep, rem, stress, deficit, intensity, restless, drift]
        # Ensure the order matches feature_names
        p_df = pd.DataFrame([p_features], columns=feature_names)
        # Handle cases where feature names or ordering in frontend might mismatch
        p_risk = float(sleep_model.predict_proba(sleep_scaler.transform(p_df[feature_names]))[0][1])

    # 4. Mental Prediction
    if text_model is not None and tokenizer is not None:
        t_inputs = tokenizer(text_msg, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            t_outputs = text_model(**t_inputs)
            t_probs = torch.nn.functional.softmax(t_outputs.logits, dim=-1).numpy()[0]
        t_risk = float(t_probs[0] + t_probs[1])

    # 5. Vocal Prediction
    if voice_model is not None and audio_file_path:
        try:
            v_input = get_voice_features(audio_file_path)
            v_risk = float(voice_model.predict(v_input, verbose=0)[0].max())
        except Exception as e:
            print(f"Error processing audio: {e}")
            v_risk = 0.0 # Fallback

    # 6. Fusion & Output
    final_score, status, advice = hybrid_fusion_logic(p_risk, t_risk, v_risk)

    return {
        "physical_score": p_risk,
        "mental_score": t_risk,
        "vocal_score": v_risk,
        "overall_score": final_score,
        "status": status,
        "advice": advice
    }
