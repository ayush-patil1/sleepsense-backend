import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "Models")

# Global model variables
feature_names = None
sleep_scaler = None
sleep_model = None

def load_models():
    """Load sleep ML model into memory."""
    global feature_names, sleep_scaler, sleep_model
    print("Initializing SleepSense AI (Sleep Model Only)...")
    
    try:
        feature_names = joblib.load(os.path.join(MODELS_PATH, 'feature_list_v3.pkl'))
        sleep_scaler = joblib.load(os.path.join(MODELS_PATH, 'sleep_scaler_v3.pkl'))
        
        sleep_model = xgb.XGBClassifier()
        sleep_model.load_model(os.path.join(MODELS_PATH, 'sleep_model_v3.json'))
        
        print("Sleep Model Loaded Successfully ✅")
    except Exception as e:
        print(f"Error loading model: {e}")


def generate_report_logic(user_input: dict):
    """
    Processes user data and returns stress prediction based only on sleep.
    """

    # 1. Extract Inputs
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

    deep = user_input.get('deep_sleep_percent', 20.0)
    rem = user_input.get('rem_sleep_percent', 22.0)

    # Auto-calculate efficiency
    eff = user_input.get('sleep_efficiency')
    if eff is None:
        in_bed_mins = (wak_m - bed_m) % 1440
        if in_bed_mins == 0:
            in_bed_mins = 480
        eff = min((dur / (in_bed_mins / 60)) * 100, 98.0)

    # 2. Feature Engineering
    deficit = 8.0 - dur
    intensity = work / (dur + 0.5)
    restless = wake * lat
    drift = min(abs(bed_m - 1380), 1440 - abs(bed_m - 1380))

    p_risk = 0.0

    # 3. Prediction
    if sleep_model is not None and sleep_scaler is not None and feature_names is not None:
        features = [
            age, gen, occ, work, dur, lat, eff, wake,
            bed_m, wak_m, deep, rem, stress,
            deficit, intensity, restless, drift
        ]

        df = pd.DataFrame([features], columns=feature_names)

        p_risk = float(
            sleep_model.predict_proba(
                sleep_scaler.transform(df[feature_names])
            )[0][1]
        )

        # Heuristic boosts
        if dur < 4:
            p_risk = max(p_risk, 0.95)
        elif dur < 5:
            p_risk = max(p_risk, 0.85)
        elif dur < 6:
            p_risk = max(p_risk, 0.60)

        if wake > 4:
            p_risk = max(p_risk, 0.70)

        if work > 12:
            p_risk = max(p_risk, 0.65)
        elif work > 10:
            p_risk = max(p_risk, 0.50)

    # 4. Final Output
    final_score = p_risk

    if final_score > 0.7:
        status = "CRITICAL RISK"
        advice = "Severe sleep issues detected. Immediate rest required."
    elif final_score > 0.4:
        status = "MODERATE RISK"
        advice = "Sleep quality is poor. Improve sleep habits."
    else:
        status = "STABLE"
        advice = "Sleep health is good."

    print(f"DEBUG | Sleep Risk: {final_score:.2f} | {status}")

    return {
        "physical_score": final_score,
        "overall_score": final_score,
        "status": status,
        "advice": advice
    }