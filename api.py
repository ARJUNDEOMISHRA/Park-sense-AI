import os, joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from bns_model import build_bnn

MODEL_DIR = ".models"
FEATURES_CYCLIC = [
    "Hour_sin","Hour_cos","DayOfWeek_sin","DayOfWeek_cos",
    "Month_sin","Month_cos","Day","IsWeekend",
    "air_temp","dew_point","wind_spd","pressure","rain_mm"
]
FEATURE_NAMES = [
    "Hour (sin)","Hour (cos)","DoW (sin)","DoW (cos)",
    "Month (sin)","Month (cos)","Day","Weekend",
    "Temperature","Dew Point","Wind Speed","Pressure","Rainfall"
]
INPUT_DIM = len(FEATURES_CYCLIC)
KL_BASE = 1.0 / 8000

# Model registry: name -> (hidden_sizes, kl_weight, weights_file)
MODEL_REGISTRY = {
    "BNS (Bayesian Neural-Symbolic)": ([64, 32], KL_BASE, "bns.weights.h5"),
    "BNN-Small (32→16)": ([32, 16], KL_BASE, "bnn_small.weights.h5"),
    "BNN-Large (128→64)": ([128, 64], KL_BASE, "bnn_large.weights.h5"),
    "BNN-LowKL (128→64)": ([128, 64], KL_BASE * 0.1, "bnn_lowkl.weights.h5"),
    "BNN-HighKL (128→64)": ([128, 64], KL_BASE * 10.0, "bnn_highkl.weights.h5"),
}

class PredictionRequest(BaseModel):
    hour: int; day: int; month: int; dow: int; is_weekend: int
    air_temp: float; dew_point: float; wind_spd: float; pressure: float; rain_mm: float
    model_choice: str = "BNS (Bayesian Neural-Symbolic)"; mc_samples: int = 50

class PredictionResponse(BaseModel):
    pred_class: int; pred_probs: List[float]
    std_probs: Optional[List[float]] = None
    mc_preds_raw: Optional[List[List[float]]] = None
    xgb_class: Optional[int] = None; bns_class: Optional[int] = None
    models_agree: Optional[bool] = None; conflict_severity: Optional[str] = None
    shap_values: Optional[List[float]] = None; shap_feature_names: Optional[List[str]] = None
    calibration_bins: Optional[List[float]] = None
    calibration_accuracy: Optional[List[float]] = None
    active_model: Optional[str] = None
    available_models: Optional[List[str]] = None
    # Data Drift
    drift_score: Optional[float] = None
    drift_status: Optional[str] = None
    drift_features: Optional[List[str]] = None

ml_models = {}
shap_explainer = None
calibration_data = None

def build_features(req):
    d = {
        "Hour_sin": np.sin(2*np.pi*req.hour/24), "Hour_cos": np.cos(2*np.pi*req.hour/24),
        "DayOfWeek_sin": np.sin(2*np.pi*req.dow/7), "DayOfWeek_cos": np.cos(2*np.pi*req.dow/7),
        "Month_sin": np.sin(2*np.pi*req.month/12), "Month_cos": np.cos(2*np.pi*req.month/12),
        "Day": req.day, "IsWeekend": req.is_weekend, "air_temp": req.air_temp,
        "dew_point": req.dew_point, "wind_spd": req.wind_spd, "pressure": req.pressure, "rain_mm": req.rain_mm
    }
    return np.array([d[f] for f in FEATURES_CYCLIC]).reshape(1,-1).astype(np.float32)

def train_and_cache_variant(name, hidden, kl_w, weights_file, scaler):
    """Train a BNN variant on synthetic data and cache weights."""
    path = os.path.join(MODEL_DIR, weights_file)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("→", "_to_").replace("-", "_")
    model = build_bnn(INPUT_DIM, 5, hidden, kl_w, safe_name)
    _ = model(np.zeros((1, INPUT_DIM), dtype=np.float32))
    
    if os.path.exists(path):
        model.load_weights(path)
        print(f"  ✓ {name}: loaded from cache")
        return model
    
    # Generate training data
    print(f"  ⏳ {name}: training...")
    rng = np.random.default_rng(42)
    n = 10000
    hours = rng.integers(0,24,n)
    X_raw = np.column_stack([
        np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
        np.sin(2*np.pi*rng.integers(0,7,n)/7), np.cos(2*np.pi*rng.integers(0,7,n)/7),
        np.sin(2*np.pi*rng.integers(1,13,n)/12), np.cos(2*np.pi*rng.integers(1,13,n)/12),
        rng.integers(1,32,n), (rng.integers(0,7,n)>=5).astype(float),
        rng.normal(17,5,n), rng.normal(17,5,n)-rng.uniform(2,8,n),
        rng.exponential(3,n), rng.normal(1013,8,n),
        rng.exponential(0.3,n)*(rng.random(n)<0.2)
    ]).astype(np.float32)
    occ = 0.5+0.3*np.sin(2*np.pi*(hours-8)/24)
    y = np.digitize(np.clip(occ+rng.normal(0,0.1,n),0,1),[0.2,0.4,0.6,0.8]).astype(np.int32)
    X_scaled = scaler.transform(X_raw)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=15, batch_size=512, verbose=0)
    model.save_weights(path)
    print(f"  ✓ {name}: trained & cached")
    return model

def compute_calibration(model, scaler, n=2000):
    rng = np.random.default_rng(99)
    hours = rng.integers(0,24,n)
    X_raw = np.column_stack([
        np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
        np.sin(2*np.pi*rng.integers(0,7,n)/7), np.cos(2*np.pi*rng.integers(0,7,n)/7),
        np.sin(2*np.pi*rng.integers(1,13,n)/12), np.cos(2*np.pi*rng.integers(1,13,n)/12),
        rng.integers(1,32,n), (rng.integers(0,7,n)>=5).astype(float),
        rng.normal(17,5,n), rng.normal(17,5,n)-rng.uniform(2,8,n),
        rng.exponential(3,n), rng.normal(1013,8,n),
        rng.exponential(0.3,n)*(rng.random(n)<0.2)
    ]).astype(np.float32)
    occ = 0.5+0.3*np.sin(2*np.pi*(hours-8)/24)
    y_true = np.digitize(np.clip(occ+rng.normal(0,0.1,n),0,1),[0.2,0.4,0.6,0.8])
    X_scaled = scaler.transform(X_raw)
    probs = model.predict_proba(X_scaled)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0,1,11)
    bin_acc, bin_conf = [], []
    for i in range(len(bins)-1):
        mask = (confidences>=bins[i])&(confidences<bins[i+1])
        if mask.sum()>0:
            bin_acc.append(float(correct[mask].mean()))
            bin_conf.append(float(confidences[mask].mean()))
        else:
            bin_acc.append(0.0); bin_conf.append((bins[i]+bins[i+1])/2)
    return {"bins": [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], "accuracy": bin_acc}

app = FastAPI(title="ParkIQ API", version="3.0.0")

@app.on_event("startup")
async def startup():
    global ml_models, shap_explainer, calibration_data
    print("═══ ParkIQ Model Zoo Loading ═══")
    scaler = joblib.load(os.path.join(MODEL_DIR,"scaler.pkl"))
    xgb_model = xgb.XGBClassifier(); xgb_model.load_model(os.path.join(MODEL_DIR,"xgb_model.json"))
    
    ml_models = {"scaler": scaler, "xgb": xgb_model, "bnn_variants": {}}
    
    # Load/train all BNN variants
    for name, (hidden, kl_w, wf) in MODEL_REGISTRY.items():
        ml_models["bnn_variants"][name] = train_and_cache_variant(name, hidden, kl_w, wf, scaler)
    
    shap_explainer = shap.TreeExplainer(xgb_model)
    calibration_data = compute_calibration(xgb_model, scaler)
    print(f"═══ {len(MODEL_REGISTRY)+1} models ready ═══")

@app.get("/health")
def health(): return {"status":"ok","models": list(MODEL_REGISTRY.keys()) + ["XGBoost (Baseline)"]}

@app.get("/models")
def list_models(): return {"models": ["XGBoost (Baseline)"] + list(MODEL_REGISTRY.keys())}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if "scaler" not in ml_models:
        raise HTTPException(503,"Models not loaded")
    x = build_features(req)
    xs = ml_models["scaler"].transform(x)
    
    all_model_names = ["XGBoost (Baseline)"] + list(MODEL_REGISTRY.keys())

    # XGBoost prediction (always, for arbitration)
    xgb_probs = ml_models["xgb"].predict_proba(xs)[0]
    xgb_class = int(np.argmax(xgb_probs))

    # Determine which BNN to use
    if req.model_choice.startswith("XGBoost"):
        probs = xgb_probs.tolist()
        pred_c = xgb_class
        std, mc_raw = None, None
        bns_class = xgb_class  # no BNS comparison
    else:
        # Find the right BNN variant
        variant_name = req.model_choice
        if variant_name not in ml_models["bnn_variants"]:
            variant_name = list(MODEL_REGISTRY.keys())[0]  # fallback to BNS
        bnn = ml_models["bnn_variants"][variant_name]
        
        mc_preds = []
        for _ in range(req.mc_samples):
            logits = bnn(xs, training=True)
            mc_preds.append(tf.nn.softmax(logits).numpy()[0])
        mc_arr = np.array(mc_preds)
        bns_probs = mc_arr.mean(axis=0)
        bns_std = mc_arr.std(axis=0)
        bns_class = int(np.argmax(bns_probs))
        probs = bns_probs.tolist()
        std = bns_std.tolist()
        pred_c = bns_class
        mc_raw = mc_arr.tolist()

    agree = xgb_class == pred_c
    diff = abs(xgb_class - pred_c)
    severity = "None" if agree else ("Minor" if diff==1 else "Major")

    # SHAP
    sv = shap_explainer.shap_values(xs)
    sv_arr = np.array(sv)
    if sv_arr.ndim == 3:
        shap_out = np.abs(sv_arr[:, 0, :]).mean(axis=0).tolist()
    elif sv_arr.ndim == 2:
        shap_out = np.abs(sv_arr[0]).tolist()
    else:
        shap_out = [0.0] * INPUT_DIM

    # Data Drift Detection (Out-of-Distribution)
    # xs contains Z-scores since StandardScaler is used
    z_scores = xs[0]
    max_z = float(np.max(np.abs(z_scores)))
    drift_feat_idx = int(np.argmax(np.abs(z_scores)))
    drift_features = [FEATURE_NAMES[drift_feat_idx]] if max_z > 2.0 else []
    
    if max_z > 3.0:
        drift_status = "Critical (Out of Distribution)"
    elif max_z > 2.0:
        drift_status = "Warning (High Deviation)"
    else:
        drift_status = "Normal"

    return PredictionResponse(
        pred_class=pred_c, pred_probs=probs, std_probs=std, mc_preds_raw=mc_raw,
        xgb_class=xgb_class, bns_class=bns_class if not req.model_choice.startswith("XGBoost") else None,
        models_agree=agree, conflict_severity=severity,
        shap_values=shap_out, shap_feature_names=FEATURE_NAMES,
        calibration_bins=calibration_data["bins"],
        calibration_accuracy=calibration_data["accuracy"],
        active_model=req.model_choice,
        available_models=all_model_names,
        drift_score=max_z,
        drift_status=drift_status,
        drift_features=drift_features
    )
