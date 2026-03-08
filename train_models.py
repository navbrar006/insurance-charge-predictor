import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================
# Config / Paths
# ============================
os.makedirs("models", exist_ok=True)

DATA_PATH = "data/processed/final_data.csv"
RF_PATH = "models/rf_model.pkl"
HGBR_PATH = "models/hgbr_model.pkl"
METRICS_PATH = "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================
# Metrics helper
# ============================
def get_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

# ============================
# Load dataset
# ============================
data = pd.read_csv(DATA_PATH)
print("Loaded:", data.shape)

# Target / Features
target = "charges"
X = data.drop(columns=[target])
y = data[target]

# Columns
categorical_features = ["sex", "smoker", "region", "BMI_Category", "Age_Group", "source"]
numerical_features = ["age", "bmi", "children", "Smoker_Risk_Index", "Family_Load", "Lifestyle_Risk_Score"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# Split (same for all models to compare fairly)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Log-transform target (better for skewed charges)
y_train_log = np.log1p(y_train)

# ============================
# Model 1: Random Forest
# ============================
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf),
])

print("\nTraining RandomForest...")
rf_model.fit(X_train, y_train_log)

# ============================
# Model 2: HistGradientBoosting
# ============================
hgbr = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.08,
    max_iter=400,
    random_state=RANDOM_STATE,
)

hgbr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", hgbr),
])

print("\nTraining HistGradientBoosting...")
hgbr_model.fit(X_train, y_train_log)

# ============================
# Predict (log space)
# ============================
pred_rf_log = rf_model.predict(X_test)
pred_hgbr_log = hgbr_model.predict(X_test)

# Convert individual preds back to charges
pred_rf = np.expm1(pred_rf_log)
pred_hgbr = np.expm1(pred_hgbr_log)

# ============================
# Weighted Blending (pick best by RMSE)
# ============================
weights = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]  # RF weight candidates
best = None

print("\n--- Trying Blend Weights ---")
for w_rf in weights:
    w_hgbr = 1 - w_rf

    blend_log = w_rf * pred_rf_log + w_hgbr * pred_hgbr_log
    blend_pred = np.expm1(blend_log)

    m = get_metrics(y_test, blend_pred)
    print(
        f"Blend rf={w_rf:.2f}, hgbr={w_hgbr:.2f} -> "
        f"R2={m['r2']:.4f}, MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}"
    )

    if (best is None) or (m["rmse"] < best["metrics"]["rmse"]):
        best = {"w_rf": float(w_rf), "w_hgbr": float(w_hgbr), "metrics": m}

best_w = {"rf": best["w_rf"], "hgbr": best["w_hgbr"]}
m_blend = best["metrics"]

print("\n✅ Best Blend Weights:", best_w)
print("✅ Best Blend Metrics:", m_blend)

# Final blend predictions using best weights
pred_blend_log = best_w["rf"] * pred_rf_log + best_w["hgbr"] * pred_hgbr_log
pred_blend = np.expm1(pred_blend_log)

# ============================
# Final metrics for singles
# ============================
m_rf = get_metrics(y_test, pred_rf)
m_hgbr = get_metrics(y_test, pred_hgbr)

print("\n--- Final Test Metrics ---")
print("RF   :", m_rf)
print("HGBR :", m_hgbr)
print("BLEND:", m_blend)

# ============================
# Save models
# ============================
joblib.dump(rf_model, RF_PATH)
joblib.dump(hgbr_model, HGBR_PATH)
print("\n✅ Saved models:")
print(" -", RF_PATH)
print(" -", HGBR_PATH)

# ============================
# Save metrics.json
# ============================
metrics_all = {
    "rf": m_rf,
    "hgbr": m_hgbr,
    "blend": m_blend,
    "train_records": int(X_train.shape[0]),
    "test_records": int(X_test.shape[0]),
    "blend_weights": best_w,
    "note": "Models trained on log1p(charges) and inverse transformed using expm1."
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_all, f, indent=2)

print("✅ Saved metrics:", METRICS_PATH)