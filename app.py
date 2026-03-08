import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit.components.v1 as components
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_build import feature_engineering

# ==============================
# Helpers: Scroll + Divider
# ==============================
def scroll_to(anchor_id: str):
    components.html(
        f"""
        <script>
        const doc = window.parent.document;

        function tryScroll() {{
            const el = doc.getElementById("{anchor_id}");
            if (el) {{
                el.scrollIntoView({{
                    behavior: "smooth",
                    block: "start"
                }});
                return true;
            }}
            return false;
        }}

        setTimeout(() => {{
            if (tryScroll()) return;

            let attempts = 0;
            const timer = setInterval(() => {{
                attempts++;
                if (tryScroll() || attempts > 20) {{
                    clearInterval(timer);
                }}
            }}, 200);
        }}, 300);
        </script>
        """,
        height=0,
    )

def sleek_divider(height=22):
    components.html(
        """
        <div style="
            height:1px;
            width:100%;
            margin:18px 0;
            background:linear-gradient(90deg,
              transparent,
              rgba(30,94,216,0.25),
              rgba(30,94,216,0.55),
              rgba(30,94,216,0.25),
              transparent
            );
            border-radius:999px;
            position:relative;
        ">
          <div style="
              position:absolute;
              left:50%;
              transform:translateX(-50%);
              top:-2px;
              height:6px;
              width:80px;
              background:#1E5ED8;
              border-radius:999px;
              opacity:.14;
          "></div>
        </div>
        """,
        height=height,
    )

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="💙",
    layout="wide",
)

# ==============================
# Paths
# ==============================
BASE_DIR = Path(__file__).resolve().parent
RF_PATH = BASE_DIR / "models" / "rf_model.pkl"
HGBR_PATH = BASE_DIR / "models" / "hgbr_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

# ==============================
# CSS (Clean + Dark Sidebar + Web-like Nav)
# ==============================
st.markdown(
    """
<style>
:root{
  --bg:#F2F6FF;
  --card:#FFFFFF;
  --text:#0B1F3A;
  --muted:#5C6B82;
  --navy:#0A2347;
  --blue:#123E7A;
  --blue2:#1E5ED8;
  --softblue:#EAF1FF;
  --border:#D8E4FF;
  --shadow:0 12px 28px rgba(10,35,71,.08);
}

html, body { background: var(--bg) !important; }
.block-container{ max-width: 1200px; padding-top: 1.1rem; padding-bottom: 2rem; }
div[data-testid="stVerticalBlock"]{ gap:.55rem !important; }

h1,h2,h3{ color: var(--text); letter-spacing: -0.3px; }
.small{ color: var(--muted); font-size: .92rem; }

.card{
  background: var(--card);
  border-radius: 18px;
  padding: 16px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
.section-title{
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 6px;
  color: var(--text);
}
.plot-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 12px;
  box-shadow: var(--shadow);
}

div[data-baseweb="input"] input{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  padding: 10px !important;
}
div[data-baseweb="select"] > div{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stMetric"]{
  background: var(--softblue);
  border-radius: 14px;
  padding: 12px;
  border: 1px solid var(--border);
}

/* HERO */
.hero{
  background:linear-gradient(135deg,#0A2347,#1E5ED8);
  border-radius:24px;
  padding:30px 34px;
  margin-bottom:18px;
  color:#fff;
  width:100%;
  overflow:hidden;
  box-shadow:0 14px 35px rgba(10,35,71,.15);
}
.hero-row{ display:flex; justify-content:space-between; align-items:center; gap:34px; }
.hero-left{ flex:1; }
.hero-right{ width:320px; }
.hero-title-row{ display:flex; align-items:center; gap:12px; }
.hero-icon{
  width:46px;height:46px;border-radius:14px;
  background:rgba(255,255,255,.15);
  display:flex;align-items:center;justify-content:center;font-size:22px;
}
.hero-title{ font-size:38px; font-weight:900; line-height:1.1; }
.hero-subtitle{ margin-top:8px; font-size:14px; color:rgba(255,255,255,.86); }
.hero-pills{ margin-top:14px; }
.pill{
  display:inline-block; padding:7px 14px; margin-right:8px; margin-bottom:8px;
  border-radius:999px; font-size:12px;
  background:rgba(255,255,255,.15);
  border:1px solid rgba(255,255,255,.25);
  color:#fff;
}
.tipbox{
  background:rgba(255,255,255,.12);
  border:1px solid rgba(255,255,255,.25);
  border-radius:16px;
  padding:16px 18px;
}
.tiphead{ font-weight:900; font-size:14px; margin-bottom:6px; }
.tipline{ font-size:13px; margin-top:6px; opacity:.9; }

/* Result card */
.result-card{
  background:linear-gradient(135deg,#0A2347,#1E5ED8);
  border-radius:22px;
  padding:28px;
  color:white;
  box-shadow:0 16px 40px rgba(0,0,0,.15);
}
.result-title{ font-size:16px; font-weight:700; opacity:.9; }
.result-value{ font-size:42px; font-weight:900; margin-top:6px; }
.result-sub{ font-size:13px; opacity:.85; margin-top:8px; }

/* Remove unwanted wrappers that create rectangles */
div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stContainer"],
div[data-testid="stMarkdownContainer"],
div.element-container,
div[data-testid="stElementContainer"]{
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
  border-radius:0 !important;
}

/* TABS (still used? keep clean) */
button[data-baseweb="tab"]{ border-radius:12px !important; }
div[data-baseweb="tab-list"]{ border-bottom:0 !important; }

/* ============ SIDEBAR DARK THEME ============ */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0A2347 0%, #081C3C 100%) !important;
  border-right: 1px solid rgba(216,228,255,.12) !important;
}
section[data-testid="stSidebar"] *{ color: rgba(255,255,255,.92) !important; }
.sidebar-title{
  font-size:16px;
  font-weight:900;
  margin: 6px 0 10px 0;
  color:#fff !important;
}
.sidebar-sub{
  font-size:12.5px;
  color: rgba(255,255,255,.75) !important;
  margin-top:2px;
}

/* Radio -> Website menu */
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap: 8px !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"]{ display:none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  width:100% !important;
  display:flex !important;
  align-items:center !important;
  gap:10px !important;
  padding:10px 12px !important;
  border-radius:12px !important;
  cursor:pointer !important;
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(216,228,255,.10) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background: rgba(255,255,255,.10) !important;
  border: 1px solid rgba(216,228,255,.18) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > div:has(input:checked) label{
  background: rgba(30,94,216,.22) !important;
  border: 1px solid rgba(30,94,216,.55) !important;
  box-shadow: 0 10px 24px rgba(0,0,0,.18) !important;
}

/* Sidebar divider */
.sidebar-divider{
  height:1px;
  width:100%;
  margin: 14px 0;
  background: linear-gradient(90deg, transparent, rgba(216,228,255,.25), transparent);
}
/* ==============================
   FIX OVERLAP / DOUBLE OUTLINES
   ============================== */

/* kill the default wrapper outlines around columns/blocks */
div[data-testid="stHorizontalBlock"]{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

div[data-testid="column"]{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding-top: 0 !important;
}

/* the invisible “element-container” that sometimes adds the outline */
div[data-testid="stElementContainer"],
div.element-container{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
}

/* add clean spacing between your actual cards only */
.card{
  margin-top: 10px !important;
}

/* profile grid card items: keep them sleek (no extra thick border feel) */
.profile-item{
  border: 1px solid rgba(216,228,255,.9) !important;
  box-shadow: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
# Loaders
# ==============================
@st.cache_resource
def load_models():
    rf = joblib.load(RF_PATH)
    hgbr = joblib.load(HGBR_PATH)
    return rf, hgbr

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

rf_model, hgbr_model = load_models()
metrics = load_metrics()

# Blend weights (from metrics.json if available)
w_rf, w_hgbr = 0.95, 0.05
if metrics and isinstance(metrics.get("blend_weights"), dict):
    w_rf = float(metrics["blend_weights"].get("rf", w_rf))
    w_hgbr = float(metrics["blend_weights"].get("hgbr", w_hgbr))

# ==============================
# Model Helpers
# ==============================
def bmi_label(b: float) -> str:
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"

def risk_level(pred: float) -> str:
    if pred < 10000:
        return "🟢 Low"
    if pred < 25000:
        return "🟡 Medium"
    return "🔴 High"

def make_input_df(age, sex, bmi, children, smoker, region):
    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "source": "user_input"
    }])
    return feature_engineering(df)

def predict_hybrid(input_df):
    prf_log = rf_model.predict(input_df)
    phg_log = hgbr_model.predict(input_df)
    blend_log = (w_rf * prf_log) + (w_hgbr * phg_log)
    pred_blend = float(np.expm1(blend_log)[0])
    pred_rf = float(np.expm1(prf_log)[0])
    pred_hg = float(np.expm1(phg_log)[0])
    return pred_blend, pred_rf, pred_hg

def simplify_feature_name(feature: str) -> str:
    f = feature.replace("preprocessor__", "").replace("cat__", "").replace("num__", "")
    if "Smoker_Risk_Index" in f: return "Smoking × BMI risk"
    if "Lifestyle_Risk_Score" in f: return "Lifestyle risk score"
    if "Family_Load" in f: return "Family load (age × children)"
    if "BMI_Category_Obese" in f: return "BMI: Obese"
    if "BMI_Category_Overweight" in f: return "BMI: Overweight"
    if "Age_Group_Senior" in f: return "Age group: Senior"
    if "Age_Group_Middle" in f: return "Age group: Middle"
    if "smoker_yes" in f: return "Smoking habit"
    if f == "bmi" or f.endswith("bmi"): return "Body Mass Index (BMI)"
    if f == "age" or f.endswith("age"): return "Age"
    if f == "children" or f.endswith("children"): return "Children"
    return f

def safe_import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception:
        return None

# ==============================
# SIDEBAR NAV
# ==============================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Health Insurance Charge Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Premium estimation + insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    nav = st.radio(
        "",
        ["🏠 Predictor & Explanation", "🧮 Cost Optimizer", "👤 Profile Summary", "ℹ️ About"],
        label_visibility="collapsed"
    )
    if "pending_scroll" not in st.session_state:
     st.session_state.pending_scroll = None

    if "prev_nav" not in st.session_state:
     st.session_state.prev_nav = nav

    if nav != st.session_state.prev_nav:
    #  if nav == "🔍 Explanation":
    #     st.session_state.pending_scroll = "explain-section"
     if nav == "🧮 Cost Optimizer":
        st.session_state.pending_scroll = "costopt-section"
    elif nav in ["🏠 Predictor", "👤 Profile Summary", "ℹ️ About"]:
        st.session_state.pending_scroll = "top"

    st.session_state.prev_nav = nav

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:12.5px; line-height:1.45; color:rgba(255,255,255,.78);">
          <b style="color:#fff;">Model:</b> Hybrid (RF + HGBR)<br/>
          Blend → RF: {w_rf:.2f}, HGBR: {w_hgbr:.2f}<br/>
          <span style="opacity:.9;">Tip:</span> Use navigation for insights.
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================
# HERO + TOP ANCHOR
# ==============================
st.markdown("<div id='top'></div>", unsafe_allow_html=True)

hero_html = """
<div class='hero'>
  <div class='hero-row'>
    <div class='hero-left'>
      <div class='hero-title-row'>
        <div class='hero-icon'>💙</div>
        <div class='hero-title'>Health Insurance Cost Predictor</div>
      </div>
      <div class='hero-subtitle'>AI-based premium estimation with transparent explanations.</div>
      <div class='hero-pills'>
        <span class='pill'>Auto BMI</span>
        <span class='pill'>Hybrid AI</span>
        <span class='pill'>Explainability</span>
        <span class='pill'>What-if</span>
      </div>
    </div>
    <div class='hero-right'>
      <div class='tipbox'>
        <div class='tiphead'>Quick tips</div>
        <div class='tipline'>• Non-smoker → usually lower premium</div>
        <div class='tipline'>• Healthy BMI → better outcomes</div>
        <div class='tipline'>• Age + BMI + smoking are top drivers</div>
      </div>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ==============================
# MAIN APP FLOW
# About/Profile = standalone pages (ONLY them)
# Else = Predictor at top + sections below with auto scroll
# ==============================

# Inputs are needed for Profile Summary too, so collect ALWAYS
st.subheader("Enter your details")
st.caption("Prediction updates automatically as you change values.")
col1, col2, col3 = st.columns([1.1, 1.1, 0.8])

with col1:
    age = st.number_input("Age", 18, 100, 30, step=1)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    region = st.selectbox("Region", ["northwest", "southeast", "southwest", "northeast"], index=0)

with col2:
    weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0, step=0.5)
    height_cm = st.number_input("Height (cm)", 120.0, 220.0, 170.0, step=0.5)
    smoker = st.selectbox("Smoker", ["no", "yes"], index=0)

with col3:
    children = st.number_input("Children", 0, 5, 0, step=1)
    height_m = height_cm / 100.0
    bmi = float(weight / (height_m ** 2)) if height_m > 0 else 0.0
    st.metric("BMI", f"{bmi:.2f}")
    st.caption(f"Category: **{bmi_label(bmi)}**")

if height_cm <= 0 or weight <= 0:
    st.error("Please enter valid height and weight.")
    st.stop()

input_df = make_input_df(age, sex, bmi, children, smoker, region)
pred_blend, pred_rf_only, pred_hg_only = predict_hybrid(input_df)

# ==============================
# STANDALONE PAGES (ONLY these)
# ==============================
if nav == "👤 Profile Summary":
    sleek_divider()

    st.markdown("""
    <style>
      .profile-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:12px;}
      .profile-item{background:#F7FAFF;border:1px solid #D8E4FF;border-radius:12px;padding:12px 14px;}
      .profile-label{font-size:12px;color:#5C6B82;}
      .profile-value{font-size:16px;font-weight:700;color:#0B1F3A;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
      <div class="card">
        <div class="section-title">Profile Summary</div>
        <div class="profile-grid">
          <div class="profile-item"><div class="profile-label">Age</div><div class="profile-value">{age} years</div></div>
          <div class="profile-item"><div class="profile-label">BMI</div><div class="profile-value">{bmi:.2f} ({bmi_label(bmi)})</div></div>
          <div class="profile-item"><div class="profile-label">Smoker</div><div class="profile-value">{smoker}</div></div>
          <div class="profile-item"><div class="profile-label">Sex</div><div class="profile-value">{sex}</div></div>
          <div class="profile-item"><div class="profile-label">Region</div><div class="profile-value">{region}</div></div>
          <div class="profile-item"><div class="profile-label">Children</div><div class="profile-value">{children}</div></div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    scroll_to("top")

elif nav == "ℹ️ About":
    sleek_divider()

    st.markdown("<div class='card'><div class='section-title'>About this app</div>", unsafe_allow_html=True)

    st.write(
        "This app predicts medical insurance charges using a **hybrid AI model** "
        "(Random Forest + Histogram Gradient Boosting). It also provides "
        "**Explainable AI insights**, **What-if analysis**, and **model evaluation** "
        "on uploaded test datasets."
    )

    st.markdown("---")

    # =========================
    # Project Overview
    # =========================
    st.subheader("Project Overview")
    st.write(
        """
        The **Insurance Charge Predictor** estimates expected insurance charges
        based on key user details such as **age, sex, BMI, number of children,
        smoking status, and region**.

        It is designed to help users understand how personal and lifestyle-related
        factors can influence medical insurance cost.
        """
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            """
            **Models Used**
            - Random Forest Regressor
            - Histogram Gradient Boosting Regressor
            - Hybrid weighted blending
            """
        )

    with info_col2:
        st.markdown(
            """
            **Main Features**
            - Insurance charge prediction
            - Explainable AI support
            - What-if scenario analysis
            - Model testing on uploaded CSV
            """
        )

    st.markdown("---")

    # =========================
    # Dataset / Inputs
    # =========================
    st.subheader("Input Features")
    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            - **Age**
            - **Sex**
            - **BMI**
            - **Children**
            """
        )

    with feature_col2:
        st.markdown(
            """
            - **Smoker**
            - **Region**
            - **Charges** *(for testing/evaluation only)*
            """
        )

    st.markdown("---")

    # =========================
    # Model Evaluation
    # =========================
    st.subheader("Model Evaluation & Testing")
    st.write("Upload a CSV file to test the trained model on unseen data.")

    uploaded_test = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        key="about_test_uploader"
    )

    st.caption("Required columns: age, sex, bmi, children, smoker, region, charges")

    with st.expander("See expected dataset format"):
        sample_df = pd.DataFrame({
            "age": [19, 18, 28],
            "sex": ["female", "male", "male"],
            "bmi": [27.90, 33.77, 33.00],
            "children": [0, 1, 3],
            "smoker": ["yes", "no", "no"],
            "region": ["southwest", "southeast", "southeast"],
            "charges": [16884.92, 1725.55, 4449.46]
        })
        st.dataframe(sample_df, use_container_width=True)

    if uploaded_test is not None:
        try:
            test_df = pd.read_csv(uploaded_test)

            st.markdown("### Uploaded Dataset")
            st.dataframe(test_df, use_container_width=True)

            required_cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
            missing_cols = [col for col in required_cols if col not in test_df.columns]

            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                test_df = test_df.copy()
                test_df = test_df.dropna(subset=required_cols)

                if len(test_df) == 0:
                    st.error("The uploaded file has no valid rows after removing missing values.")
                else:
                    y_true = pd.to_numeric(test_df["charges"], errors="coerce")
                    X_test_raw = test_df.drop(columns=["charges"]).copy()

                    # Fix for feature_engineering expecting 'source'
                    if "source" not in X_test_raw.columns:
                        X_test_raw["source"] = "uploaded_test"

                    valid_mask = y_true.notna()
                    X_test_raw = X_test_raw.loc[valid_mask].reset_index(drop=True)
                    y_true = y_true.loc[valid_mask].reset_index(drop=True)

                    if len(y_true) == 0:
                        st.error("No valid numeric values found in 'charges' column.")
                    else:
                        X_test = feature_engineering(X_test_raw.copy())
                        aligned_idx = X_test.index

                        X_test_raw = X_test_raw.loc[aligned_idx].reset_index(drop=True)
                        y_true = y_true.loc[aligned_idx].reset_index(drop=True)
                        X_test = X_test.reset_index(drop=True)

                        rf_model = joblib.load(RF_PATH)
                        hgbr_model = joblib.load(HGBR_PATH)

    
                        rf_pred_log = rf_model.predict(X_test)
                        hgbr_pred_log = hgbr_model.predict(X_test)
                        rf_pred = np.expm1(rf_pred_log)
                        hgbr_pred = np.expm1(hgbr_pred_log)
                        hybrid_pred = np.expm1((w_rf * rf_pred_log) + (w_hgbr * hgbr_pred_log))
    

                        selected_model = st.selectbox(
                           "Choose prediction model",
                           ["Hybrid Model", "Random Forest", "Histogram Gradient Boosting"],
                           key="about_model_selector"
                        )

                        if selected_model == "Random Forest":
                          y_pred = rf_pred
                        elif selected_model == "Histogram Gradient Boosting":
                          y_pred = hgbr_pred
                        else:
                          y_pred = hybrid_pred

                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_true, y_pred)

                        non_zero_mask = y_true != 0
                        if np.sum(non_zero_mask) > 0:
                         mape = np.mean(
                         np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
                         ) * 100
                         approx_accuracy = max(0.0, 100 - mape)
                        else:
                         mape = np.nan
                         approx_accuracy = np.nan

                        residuals = y_true - y_pred
                        abs_error = np.abs(residuals)

                        st.markdown("### Evaluation Summary")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("R² Score", f"{r2:.3f}")
                        m2.metric("MAE", f"{mae:,.2f}")
                        m3.metric("RMSE", f"{rmse:,.2f}")
                        m4.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")

                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Rows Tested", len(y_true))
                        s2.metric("Avg Actual", f"{np.mean(y_true):,.2f}")
                        s3.metric("Avg Predicted", f"{np.mean(y_pred):,.2f}")
                        s4.metric(
                          "Approx Accuracy",
                          f"{approx_accuracy:.2f}%" if not np.isnan(approx_accuracy) else "N/A"
                        )

                        st.success(f"Model explains about {r2 * 100:.1f}% variance in insurance charges.")

                        st.caption(
                        "For regression models, R², MAE, and RMSE are the main evaluation metrics. "
                        "Approx Accuracy is shown only as a simple demo indicator."
                        )

                        results_df = X_test_raw.copy()
                        results_df["Actual Charges"] = y_true.values
                        results_df["RF Prediction"] = rf_pred
                        results_df["HGBR Prediction"] = hgbr_pred
                        results_df["Hybrid Prediction"] = hybrid_pred
                        results_df["Selected Prediction"] = y_pred
                        results_df["Residual"] = residuals
                        results_df["Absolute Error"] = abs_error

                        display_df = results_df.drop(columns=["source"], errors="ignore")

                        st.markdown("### Actual vs Predicted Results")
                        st.dataframe(display_df, use_container_width=True)
                        
                        csv = display_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Evaluation Results",
                            data=csv,
                            file_name="insurance_model_evaluation_results.csv",
                            mime="text/csv"
                        )

                        st.markdown("### Visual Evaluation")
                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            st.markdown("#### Actual vs Predicted")
                            fig1, ax1 = plt.subplots(figsize=(7, 5))
                            ax1.scatter(y_true, y_pred, alpha=0.7)
                            min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
                            max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
                            ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--")
                            ax1.set_xlabel("Actual Charges")
                            ax1.set_ylabel("Predicted Charges")
                            ax1.set_title(f"Actual vs Predicted ({selected_model})")
                            st.pyplot(fig1)

                        with chart_col2:
                            st.markdown("#### Residual Plot")
                            fig2, ax2 = plt.subplots(figsize=(7, 5))
                            ax2.scatter(y_pred, residuals, alpha=0.7)
                            ax2.axhline(y=0, linestyle="--")
                            ax2.set_xlabel("Predicted Charges")
                            ax2.set_ylabel("Residuals")
                            ax2.set_title("Residual Plot")
                            st.pyplot(fig2)

                        st.markdown("### Error Analysis")
                        best_col, worst_col = st.columns(2)

                        with best_col:
                            st.markdown("#### Best Predictions")
                            st.dataframe(
                                display_df.sort_values("Absolute Error").head(5),
                                use_container_width=True
                            )

                        with worst_col:
                            st.markdown("#### Worst Predictions")
                            st.dataframe(
                                display_df.sort_values("Absolute Error", ascending=False).head(5),
                                use_container_width=True
                            )

                        with st.expander("Technical metrics (optional)"):
                            st.write(f"**Selected model:** {selected_model}")
                            st.write(f"**Blend weights →** RF: `{w_rf:.2f}`, HGBR: `{w_hgbr:.2f}`")

                            if metrics:
                                st.write("**Stored metrics from metrics.json:**")
                                if "blend" in metrics:
                                    st.write(f"Hybrid R²: **{metrics['blend'].get('r2', 0):.3f}**")
                                    st.write(f"Hybrid RMSE: **{metrics['blend'].get('rmse', 0):.0f}**")
                                    st.write(f"Hybrid MAE: **{metrics['blend'].get('mae', 0):.0f}**")
                                else:
                                    st.warning("`blend` key not found in metrics.json")
                            else:
                                st.warning("metrics.json not found.")

        except Exception as e:
            st.error(f"Error while evaluating model: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    scroll_to("top")
# ==============================
# MAIN PREDICTOR PAGE
# (Predictor + Explanation + Cost Optimizer)
# ==============================
else:
    sleek_divider()

    # ---- Predictor cards (always visible for Predictor/Explanation/Cost Optimizer)
    out1, out2, out3 = st.columns([1.15, 0.9, 0.95])

    with out1:
        st.markdown(
            f"""
            <div class="card">
              <div class="section-title">Estimated insurance charges</div>
              <div class="result-card">
                <div class="result-title">Estimated Insurance Premium</div>
                <div class="result-value">₹ {pred_blend:,.0f}</div>
                <div class="result-sub">AI predicted premium based on your profile</div>
              </div>
              <div class="small" style="margin-top:10px;">
                This is an estimate generated from patterns learned from historical data.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with out2:
        st.markdown('<div class="card"><div class="section-title">Risk level</div></div>', unsafe_allow_html=True)
        st.metric("Risk", risk_level(pred_blend))
        if smoker == "yes":
            st.info("Stopping smoking can reduce expected premium significantly.")
        elif bmi >= 30:
            st.info("Reducing BMI can improve expected premium.")
        else:
            st.info("Maintaining healthy habits helps keep costs stable.")

    with out3:
        st.markdown('<div class="card"><div class="section-title">Simple explanation</div></div>', unsafe_allow_html=True)
        points = []
        if smoker == "yes":
            points.append("Smoking is one of the biggest cost drivers.")
        if bmi >= 30:
            points.append("High BMI increases health-risk related costs.")
        if age >= 50:
            points.append("Premiums often increase with age.")
        if children >= 3:
            points.append("More dependents can slightly increase costs.")
        if not points:
            points.append("Your inputs look relatively low-risk compared to typical patterns.")
        for p in points[:4]:
            st.write("• " + p)
        st.caption("Note: This is not medical advice.")

    # ---- Explanation section
    sleek_divider()
    st.markdown("<div id='explain-section' style='height: 1px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Top reasons behind your estimate")
        st.caption("We show the most influential factors and their contribution to your predicted premium.")

        shap = safe_import_shap()
        if shap is None:
            st.warning("SHAP is not installed. Add it to requirements.txt as: `shap` (or `shap==0.46.0`).")
        else:
            try:
                import plotly.express as px
            except Exception:
                px = None
                st.warning("Plotly not installed. Add to requirements.txt: `plotly`.")

            preprocessor = rf_model.named_steps["preprocessor"]
            tree = rf_model.named_steps["model"]

            X_transformed = preprocessor.transform(input_df)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            feature_names = preprocessor.get_feature_names_out()

            with st.spinner("Generating explanation..."):
                explainer = shap.TreeExplainer(tree)
                shap_values = explainer.shap_values(X_transformed)

            shap_series = pd.Series(shap_values[0], index=feature_names)
            abs_vals = shap_series.abs()

            if abs_vals.sum() == 0:
                st.info("Explanation not available for this input.")
            else:
                shap_percent = (abs_vals / abs_vals.sum()) * 100
                topk = 6
                top = shap_percent.sort_values(ascending=False).head(topk)

                top_signed = shap_series.loc[top.index]
                direction = np.where(top_signed.values >= 0, "⬆️ Increases", "⬇️ Decreases")

                table = pd.DataFrame({
                    "Factor": [simplify_feature_name(f) for f in top.index],
                    "Contribution (%)": top.values.round(2),
                    "Impact": direction
                })

                st.dataframe(table, use_container_width=True)
                st.success(
                    f"Top driver: **{table.iloc[0]['Factor']}** "
                    f"(**{table.iloc[0]['Contribution (%)']}%**, {table.iloc[0]['Impact']})"
                )

                if px is not None:
                    if "show_explain_charts" not in st.session_state:
                        st.session_state.show_explain_charts = False

                    btn_label = "📊 Show charts" if not st.session_state.show_explain_charts else "❌ Hide charts"
                    if st.button(btn_label, use_container_width=True):
                        st.session_state.show_explain_charts = not st.session_state.show_explain_charts

                    if st.session_state.show_explain_charts:
                        chart_type = st.radio("Choose chart style", ["Donut", "Bar"], horizontal=True)
                        chart_df = table.copy()
                        chart_df["Contribution (%)"] = chart_df["Contribution (%)"].astype(float)

                        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                        if chart_type == "Donut":
                            fig = px.pie(chart_df, names="Factor", values="Contribution (%)", hole=0.55)
                            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            bar_df = chart_df.sort_values("Contribution (%)", ascending=True)
                            fig = px.bar(bar_df, x="Contribution (%)", y="Factor", orientation="h", text="Contribution (%)")
                            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Cost optimizer (ONLY when clicked)
    if nav == "🧮 Cost Optimizer":
        sleek_divider()
        st.markdown("<div id='costopt-section' style='height: 1px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("Cost Optimizer (What-if)")
            st.caption("These are model-based estimates (not medical advice).")

            cA, cB = st.columns(2)
            with cA:
                df_non_smoker = make_input_df(age, sex, bmi, children, "no", region)
                pred_ns, _, _ = predict_hybrid(df_non_smoker)
                st.metric("If you stop smoking", f"₹ {pred_ns:,.2f}", f"₹ {pred_ns - pred_blend:,.2f}")

            with cB:
                bmi_reduced = max(12.0, bmi - 2.0)
                df_bmi_reduce = make_input_df(age, sex, bmi_reduced, children, smoker, region)
                pred_br, _, _ = predict_hybrid(df_bmi_reduce)
                st.metric("If BMI reduces by 2", f"₹ {pred_br:,.2f}", f"₹ {pred_br - pred_blend:,.2f}")
    if st.session_state.pending_scroll:
           scroll_to(st.session_state.pending_scroll)
           st.session_state.pending_scroll = None
        #     # ---- Auto-scroll triggers
        # if nav == "🔍 Explanation":
        #   scroll_to("explain-section")
        # elif nav == "🧮 Cost Optimizer":
        #   scroll_to("costopt-section")
        # else:
        #   scroll_to("top")
