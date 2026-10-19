import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# 1. PAGE CONFIGURATION & STYLES
# =====================================================
st.set_page_config(
    page_title="ChurnIQ — Customer Churn Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    /* Global styles */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card styling */
    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .app-header p {
        margin: 8px 0 0 0;
        font-size: 1.05rem;
        color: #94a3b8;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }
    
    /* Risk Badges */
    .risk-badge-high {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    .risk-badge-medium {
        background-color: #fef3c7;
        color: #b45309;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #fcd34d;
    }
    .risk-badge-low {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    
    /* Risk Drivers & Recommendations boxes */
    .driver-box {
        background-color: #f8fafc;
        border-left: 4px solid #ef4444;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .strength-box {
        background-color: #f8fafc;
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .rec-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. MODEL & ARTIFACT PATHS & CONSTANTS
# =====================================================
ARTIFACT_DIR = "models"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "tuned_xgboost.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.pkl")
XGB_THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "xgboost_threshold.pkl")

TENURE_GROUPS = [
    "1 - 12",
    "13 - 24",
    "25 - 36",
    "37 - 48",
    "49 - 60",
    "61 - 72",
]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


# =====================================================
# 3. LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file missing: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Load tuned threshold (default from threshold.pkl is 0.30)
    tuned_threshold = 0.30
    if os.path.exists(THRESHOLD_PATH):
        raw_val = joblib.load(THRESHOLD_PATH)
        if isinstance(raw_val, dict):
            tuned_threshold = float(raw_val.get("threshold", 0.30))
        else:
            tuned_threshold = float(raw_val)
    elif os.path.exists(XGB_THRESHOLD_PATH):
        raw_val = joblib.load(XGB_THRESHOLD_PATH)
        if isinstance(raw_val, dict):
            tuned_threshold = float(raw_val.get("threshold", 0.30))

    if hasattr(preprocessor, "feature_names_in_"):
        feature_columns = list(preprocessor.feature_names_in_)
    else:
        feature_columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
            "TotalCharges", "tenure_group",
        ]

    return model, preprocessor, tuned_threshold, feature_columns


try:
    model, preprocessor, default_threshold, FEATURE_COLUMNS = load_artifacts()
except Exception as exc:
    st.error(f"⚠️ Critical error loading model artifacts: {exc}")
    st.stop()


# =====================================================
# 4. ROBUST PREPROCESSING PIPELINE
# =====================================================
def derive_tenure_group(tenure_series):
    """Derives tenure_group bins exactly matching the training notebook."""
    clipped = pd.to_numeric(tenure_series, errors="coerce").fillna(1).clip(lower=1, upper=72)
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    return pd.cut(clipped, bins=range(1, 80, 12), right=False, labels=labels).astype(str)


def clean_and_prepare_input(df):
    """
    Cleans raw uploaded customer data or single customer form data.
    - Handles case-insensitive column names
    - Derives tenure_group if only tenure is provided
    - Converts SeniorCitizen (0/1 or string) to categorical 'No'/'Yes'
    - Coerces and imputes numeric values safely
    - Preserves customer identifiers for downstream tracking
    """
    clean_df = df.copy()
    clean_df.columns = [str(c).strip() for c in clean_df.columns]

    # Find customer identifier if available
    id_series = None
    for col in clean_df.columns:
        if col.lower() in ["customerid", "id", "customer_id", "cust_id", "custid"]:
            id_series = clean_df[col].copy()
            break

    # Standardize column casing for expected features
    col_map = {}
    lower_to_feature = {c.lower(): c for c in FEATURE_COLUMNS + ["tenure"]}
    for c in clean_df.columns:
        c_low = c.lower()
        if c_low in lower_to_feature and c != lower_to_feature[c_low]:
            col_map[c] = lower_to_feature[c_low]
    if col_map:
        clean_df.rename(columns=col_map, inplace=True)

    # 1. Handle tenure & tenure_group
    if "tenure_group" not in clean_df.columns:
        if "tenure" in clean_df.columns:
            clean_df["tenure_group"] = derive_tenure_group(clean_df["tenure"])
        else:
            clean_df["tenure_group"] = "1 - 12"
    else:
        clean_df["tenure_group"] = clean_df["tenure_group"].astype(str).str.strip()
        invalid_mask = ~clean_df["tenure_group"].isin(TENURE_GROUPS)
        if invalid_mask.any():
            if "tenure" in clean_df.columns:
                clean_df.loc[invalid_mask, "tenure_group"] = derive_tenure_group(
                    clean_df.loc[invalid_mask, "tenure"]
                )
            else:
                clean_df.loc[invalid_mask, "tenure_group"] = "1 - 12"

    # 2. Handle SeniorCitizen (Preprocessors expects categorical strings 'No' or 'Yes')
    if "SeniorCitizen" in clean_df.columns:
        def map_senior(val):
            val_str = str(val).strip().lower()
            if val_str in ["1", "1.0", "yes", "true", "y"]:
                return "Yes"
            return "No"
        clean_df["SeniorCitizen"] = clean_df["SeniorCitizen"].apply(map_senior)
    else:
        clean_df["SeniorCitizen"] = "No"

    # 3. Numeric columns: MonthlyCharges, TotalCharges
    for col in ["MonthlyCharges", "TotalCharges"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        else:
            clean_df[col] = 0.0

    # Safely impute missing TotalCharges
    if "MonthlyCharges" in clean_df.columns:
        if "tenure" in clean_df.columns:
            tenure_vals = pd.to_numeric(clean_df["tenure"], errors="coerce").fillna(1).clip(lower=1)
            clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(
                clean_df["MonthlyCharges"] * tenure_vals
            )
        else:
            clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(clean_df["MonthlyCharges"])
    clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(0.0)
    clean_df["MonthlyCharges"] = clean_df["MonthlyCharges"].fillna(clean_df["TotalCharges"])

    # 4. Fill missing categorical values with known defaults
    cat_steps = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = list(preprocessor.transformers_[1][2])
    for col, cats in zip(cat_cols, cat_steps.categories_):
        default_val = cats[0]
        if col not in clean_df.columns:
            clean_df[col] = default_val
        else:
            clean_df[col] = clean_df[col].astype(str).str.strip()
            clean_df.loc[clean_df[col].isin(["nan", "None", "", "NaN", "null"]), col] = default_val

    # Strict feature extraction
    features_df = clean_df[FEATURE_COLUMNS].copy()
    return features_df, clean_df, id_series


def predict_customers(raw_df, decision_threshold):
    """
    Runs model inference with the specified decision threshold.
    Returns clean dataframe with predictions and risk levels.
    """
    features_df, clean_df, id_series = clean_and_prepare_input(raw_df)
    transformed_data = preprocessor.transform(features_df)
    probabilities = model.predict_proba(transformed_data)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)

    results = clean_df.copy()
    if id_series is not None:
        if "customerID" in results.columns:
            results.drop(columns=["customerID"], inplace=True)
        if "id" in results.columns:
            results.drop(columns=["id"], inplace=True)
        results.insert(0, "Customer_ID", id_series.values)

    results["Churn Probability (%)"] = np.round(probabilities * 100, 2)
    results["Prediction"] = np.where(predictions == 1, "Likely to Churn", "Unlikely to Churn")
    results["Risk Level"] = np.where(
        predictions == 0,
        "Low",
        np.where(probabilities >= 0.70, "High", "Medium")
    )

    return results, probabilities, predictions


def get_driver_analysis(row):
    """Identifies the customer's key risk factors and retention strengths based on notebook EDA."""
    risk_factors = []
    strengths = []

    # Contract
    if row.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract (Highest churn risk ~42.7%)")
    elif row.get("Contract") in ["One year", "Two year"]:
        strengths.append(f"Long-term commitment: {row.get('Contract')} contract")

    # Tenure
    tg = str(row.get("tenure_group", ""))
    if "1 - 12" in tg:
        risk_factors.append("New customer within the first 12 months (Peak attrition window)")
    elif any(g in tg for g in ["49 - 60", "61 - 72"]):
        strengths.append("Established customer tenure (> 4 years loyalty)")

    # Internet Service
    if row.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic subscription (High historical churn without support services)")
    elif row.get("InternetService") == "No":
        strengths.append("Basic utility user (Low churn rate)")

    # Tech Support & Security
    if row.get("TechSupport") == "No":
        risk_factors.append("No Tech Support subscription")
    elif row.get("TechSupport") == "Yes":
        strengths.append("Subscribed to Tech Support (Improves retention)")

    if row.get("OnlineSecurity") == "No":
        risk_factors.append("No Online Security protection")
    elif row.get("OnlineSecurity") == "Yes":
        strengths.append("Subscribed to Online Security")

    # Payment Method
    if row.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Payment via Electronic Check (Associated with ~45% churn rate)")
    elif "automatic" in str(row.get("PaymentMethod", "")):
        strengths.append("Enrolled in Automatic Payment (Credit card / Bank transfer)")

    # Senior Citizen
    if row.get("SeniorCitizen") == "Yes":
        risk_factors.append("Senior citizen demographic (Elevated churn sensitivity)")

    # Partner / Dependents
    if row.get("Partner") == "Yes" or row.get("Dependents") == "Yes":
        strengths.append("Family / multi-user household (Higher switching costs)")

    return risk_factors, strengths




# =====================================================
# 5. SIDEBAR: MODEL CONTROLS & THRESHOLD
# =====================================================
with st.sidebar:
    st.markdown("## 🔮 **ChurnIQ Engine**")
    st.caption("AI-Powered Retention Intelligence")
    st.divider()

    st.markdown("### ⚙️ **Decision Threshold**")
    st.write(
        "Adjust classification sensitivity. The notebook analysis optimized the threshold "
        "to **0.30**, which captures **76.5% of churners** (compared to only 53.2% at 0.50)."
    )

    preset = st.radio(
        "Threshold Strategy:",
        ["Tuned (0.30 - High Recall)", "Balanced (0.40)", "Conservative (0.50)", "Custom"],
        index=0,
    )

    if preset == "Tuned (0.30 - High Recall)":
        decision_threshold = 0.30
    elif preset == "Balanced (0.40)":
        decision_threshold = 0.40
    elif preset == "Conservative (0.50)":
        decision_threshold = 0.50
    else:
        decision_threshold = st.slider(
            "Custom Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(default_threshold),
            step=0.01,
        )

    st.info(f"**Active Threshold:** `{decision_threshold:.2f}`\n\nCustomers with probability ≥ {decision_threshold*100:.0f}% will be flagged as **Likely to Churn**.")

    st.divider()
    st.markdown("### 📊 **Model Architecture**")
    st.markdown(
        """
        - **Model:** Tuned XGBoost Classifier
        - **Preprocessing:** StandardScaler + OneHotEncoder
        - **Validation ROC-AUC:** `0.838`
        - **Recall @ 0.30 Threshold:** `76.5%`
        - **Feature Dimension:** 51 Encoded Features
        """
    )
    st.divider()
    st.caption("ChurnIQ v2.0 | Built with Streamlit & XGBoost")


# =====================================================
# 6. APP HEADER
# =====================================================
st.markdown("""
<div class="app-header">
    <h1>ChurnIQ — Customer Churn Prediction System</h1>
    <p>Predict customer attrition, diagnose key churn risk factors, and deploy proactive retention interventions.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. NAVIGATION TABS
# =====================================================
tab_single, tab_batch, tab_insights = st.tabs([
    "👤 Single Customer Prediction",
    "📁 Batch CSV Prediction",
    "📈 Model & Notebook Insights",
])


# =====================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# =====================================================
with tab_single:
    st.markdown("### Evaluate Individual Customer Risk")
    st.write("Fill in customer details below or select a quick demo profile to see real-time predictions.")

    # Demo Profile Presets
    demo_cols = st.columns([1, 1, 2])
    with demo_cols[0]:
        if st.button("⚡ Load High Churn Risk Demo", use_container_width=True):
            st.session_state["demo_gender"] = "Female"
            st.session_state["demo_senior"] = "Yes"
            st.session_state["demo_partner"] = "No"
            st.session_state["demo_dependents"] = "No"
            st.session_state["demo_tenure"] = 3
            st.session_state["demo_phone"] = "Yes"
            st.session_state["demo_lines"] = "Yes"
            st.session_state["demo_internet"] = "Fiber optic"
            st.session_state["demo_security"] = "No"
            st.session_state["demo_backup"] = "No"
            st.session_state["demo_protection"] = "No"
            st.session_state["demo_support"] = "No"
            st.session_state["demo_tv"] = "Yes"
            st.session_state["demo_movies"] = "Yes"
            st.session_state["demo_contract"] = "Month-to-month"
            st.session_state["demo_paperless"] = "Yes"
            st.session_state["demo_payment"] = "Electronic check"
            st.session_state["demo_monthly"] = 98.50
            st.session_state["demo_total"] = 295.50

    with demo_cols[1]:
        if st.button("🛡️ Load Low Churn Risk Demo", use_container_width=True):
            st.session_state["demo_gender"] = "Male"
            st.session_state["demo_senior"] = "No"
            st.session_state["demo_partner"] = "Yes"
            st.session_state["demo_dependents"] = "Yes"
            st.session_state["demo_tenure"] = 48
            st.session_state["demo_phone"] = "Yes"
            st.session_state["demo_lines"] = "No"
            st.session_state["demo_internet"] = "DSL"
            st.session_state["demo_security"] = "Yes"
            st.session_state["demo_backup"] = "Yes"
            st.session_state["demo_protection"] = "Yes"
            st.session_state["demo_support"] = "Yes"
            st.session_state["demo_tv"] = "No"
            st.session_state["demo_movies"] = "No"
            st.session_state["demo_contract"] = "Two year"
            st.session_state["demo_paperless"] = "No"
            st.session_state["demo_payment"] = "Credit card (automatic)"
            st.session_state["demo_monthly"] = 55.00
            st.session_state["demo_total"] = 2640.00

    # Input Form
    with st.form("customer_input_form"):
        col_sec1, col_sec2, col_sec3 = st.columns(3)

        # SECTION 1: DEMOGRAPHICS & ACCOUNT
        with col_sec1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox(
                "Gender",
                CATEGORICAL_OPTIONS["gender"],
                index=0 if st.session_state.get("demo_gender", "Female") == "Female" else 1,
            )
            senior = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                index=1 if st.session_state.get("demo_senior", "No") == "Yes" else 0,
                help="Customer aged 65 or older. Preprocessor strictly expects 'No' or 'Yes'.",
            )
            partner = st.selectbox(
                "Has Partner?",
                CATEGORICAL_OPTIONS["Partner"],
                index=0 if st.session_state.get("demo_partner", "Yes") == "Yes" else 1,
            )
            dependents = st.selectbox(
                "Has Dependents?",
                CATEGORICAL_OPTIONS["Dependents"],
                index=1 if st.session_state.get("demo_dependents", "No") == "No" else 0,
            )

            st.markdown("#### 📅 Tenure & Lifecycle")
            tenure_months = st.slider(
                "Tenure (Months)",
                min_value=1,
                max_value=72,
                value=int(st.session_state.get("demo_tenure", 12)),
                help="Duration of customer subscription in months.",
            )
            # Automatic derivation of tenure group
            derived_group = derive_tenure_group(pd.Series([tenure_months])).iloc[0]
            st.info(f"Derived Tenure Group: **{derived_group}**")

        # SECTION 2: SERVICES & CONNECTIONS
        with col_sec2:
            st.markdown("#### 🌐 Telecom & Internet")
            phone = st.selectbox(
                "Phone Service",
                CATEGORICAL_OPTIONS["PhoneService"],
                index=0 if st.session_state.get("demo_phone", "Yes") == "Yes" else 1,
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                CATEGORICAL_OPTIONS["MultipleLines"],
                index=CATEGORICAL_OPTIONS["MultipleLines"].index(
                    st.session_state.get("demo_lines", "No")
                ),
            )
            internet = st.selectbox(
                "Internet Service",
                CATEGORICAL_OPTIONS["InternetService"],
                index=CATEGORICAL_OPTIONS["InternetService"].index(
                    st.session_state.get("demo_internet", "DSL")
                ),
            )
            security = st.selectbox(
                "Online Security",
                CATEGORICAL_OPTIONS["OnlineSecurity"],
                index=CATEGORICAL_OPTIONS["OnlineSecurity"].index(
                    st.session_state.get("demo_security", "No")
                ),
            )
            backup = st.selectbox(
                "Online Backup",
                CATEGORICAL_OPTIONS["OnlineBackup"],
                index=CATEGORICAL_OPTIONS["OnlineBackup"].index(
                    st.session_state.get("demo_backup", "Yes")
                ),
            )
            protection = st.selectbox(
                "Device Protection",
                CATEGORICAL_OPTIONS["DeviceProtection"],
                index=CATEGORICAL_OPTIONS["DeviceProtection"].index(
                    st.session_state.get("demo_protection", "No")
                ),
            )
            support = st.selectbox(
                "Tech Support",
                CATEGORICAL_OPTIONS["TechSupport"],
                index=CATEGORICAL_OPTIONS["TechSupport"].index(
                    st.session_state.get("demo_support", "No")
                ),
            )

        # SECTION 3: CONTRACT & BILLING
        with col_sec3:
            st.markdown("#### 📺 Entertainment")
            streaming_tv = st.selectbox(
                "Streaming TV",
                CATEGORICAL_OPTIONS["StreamingTV"],
                index=CATEGORICAL_OPTIONS["StreamingTV"].index(
                    st.session_state.get("demo_tv", "No")
                ),
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                CATEGORICAL_OPTIONS["StreamingMovies"],
                index=CATEGORICAL_OPTIONS["StreamingMovies"].index(
                    st.session_state.get("demo_movies", "No")
                ),
            )

            st.markdown("#### 💳 Contract & Billing")
            contract = st.selectbox(
                "Contract Type",
                CATEGORICAL_OPTIONS["Contract"],
                index=CATEGORICAL_OPTIONS["Contract"].index(
                    st.session_state.get("demo_contract", "Month-to-month")
                ),
            )
            paperless = st.selectbox(
                "Paperless Billing",
                CATEGORICAL_OPTIONS["PaperlessBilling"],
                index=CATEGORICAL_OPTIONS["PaperlessBilling"].index(
                    st.session_state.get("demo_paperless", "Yes")
                ),
            )
            payment = st.selectbox(
                "Payment Method",
                CATEGORICAL_OPTIONS["PaymentMethod"],
                index=CATEGORICAL_OPTIONS["PaymentMethod"].index(
                    st.session_state.get("demo_payment", "Electronic check")
                ),
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.get("demo_monthly", 70.0)),
                step=1.0,
            )
            auto_total = round(monthly_charges * tenure_months, 2)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=15000.0,
                value=float(st.session_state.get("demo_total", auto_total)),
                step=10.0,
                help=f"Default calculated as MonthlyCharges × Tenure = ${auto_total:.2f}",
            )

        submit_single = st.form_submit_button(
            "🚀 Predict Churn Probability",
            type="primary",
            use_container_width=True,
        )

    # Prediction Output
    if submit_single:
        customer_record = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure_months,
            "tenure_group": derived_group,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        try:
            results_df, probs, preds = predict_customers(customer_record, decision_threshold)
            result = results_df.iloc[0]
            prob_val = float(probs[0])
            churn_perc = float(result["Churn Probability (%)"])
            risk_lvl = result["Risk Level"]
            prediction_str = result["Prediction"]

            st.divider()
            st.markdown("### 🎯 **Prediction Results & Risk Assessment**")

            # Risk Banner
            if risk_lvl == "High":
                st.error(f"🚨 **HIGH RISK CUSTOMER DETECTED**: Churn probability is **{churn_perc:.1f}%**. Proactive retention intervention strongly advised.")
            elif risk_lvl == "Medium":
                st.warning(f"⚠️ **MEDIUM RISK CUSTOMER DETECTED**: Churn probability is **{churn_perc:.1f}%** (exceeds {decision_threshold*100:.0f}% threshold). Monitor customer closely.")
            else:
                st.success(f"✅ **LOW RISK CUSTOMER**: Churn probability is **{churn_perc:.1f}%**. Customer shows high stability and retention affinity.")

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Status", prediction_str)
            m2.metric("Churn Probability", f"{churn_perc:.1f}%")
            m3.metric("Risk Classification", risk_lvl)
            m4.metric("Active Threshold", f"{decision_threshold:.2f}")

            # Visual Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_perc,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#ef4444" if risk_lvl == "High" else "#f59e0b" if risk_lvl == "Medium" else "#10b981"},
                    'steps': [
                        {'range': [0, decision_threshold * 100], 'color': "rgba(16, 185, 129, 0.15)"},
                        {'range': [decision_threshold * 100, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': decision_threshold * 100,
                    }
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as exc:
            st.error(f"❌ Prediction computation error: {exc}")


# =====================================================
# TAB 2: BATCH CSV PREDICTION
# =====================================================
with tab_batch:
    st.markdown("### Batch Customer Churn Inference")
    st.write(
        "Upload a customer dataset in CSV format. The system automatically handles raw columns, "
        "converts `SeniorCitizen`, derives `tenure_group` from `tenure`, imputes blank `TotalCharges`, "
        "and preserves customer identifiers."
    )

    batch_btn_col1, batch_btn_col2 = st.columns([1, 2])
    with batch_btn_col1:
        load_sample = st.button("📂 Load Kaggle Competition Sample (200 records)", use_container_width=True)

    uploaded_csv = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        key="batch_uploader",
    )

    batch_data = None
    if load_sample:
        kaggle_test_path = os.path.join("Dataset", "kaggle_competition_dataset", "test.csv")
        if os.path.exists(kaggle_test_path):
            batch_data = pd.read_csv(kaggle_test_path, nrows=200)
            st.success("Loaded 200 sample customer records from Kaggle test set.")
        else:
            telco_path = os.path.join("Dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
            if os.path.exists(telco_path):
                batch_data = pd.read_csv(telco_path, nrows=200)
                st.success("Loaded 200 sample customer records from Telco dataset.")

    elif uploaded_csv is not None:
        try:
            batch_data = pd.read_csv(uploaded_csv)
            st.success(f"Uploaded CSV with {len(batch_data):,} customer records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if batch_data is not None:
        st.markdown("#### Input Data Preview")
        st.dataframe(batch_data.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Churn Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing batch data with XGBoost model..."):
                try:
                    pred_results, b_probs, b_preds = predict_customers(batch_data, decision_threshold)

                    total_customers = len(pred_results)
                    churners_count = int(b_preds.sum())
                    churn_rate = churners_count / total_customers if total_customers > 0 else 0
                    high_risk = int((pred_results["Risk Level"] == "High").sum())
                    med_risk = int((pred_results["Risk Level"] == "Medium").sum())
                    low_risk = int((pred_results["Risk Level"] == "Low").sum())

                    # Revenue at risk
                    rev_at_risk = 0.0
                    if "MonthlyCharges" in pred_results.columns:
                        rev_at_risk = float(pred_results.loc[pred_results["Prediction"] == "Likely to Churn", "MonthlyCharges"].sum())

                    st.divider()
                    st.markdown("### 📊 **Executive Summary Dashboard**")

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Customers Analyzed", f"{total_customers:,}")
                    k2.metric("Predicted Churners", f"{churners_count:,}", f"{churn_rate:.1%} Churn Rate")
                    k3.metric("High Risk Accounts", f"{high_risk:,}", f"{(high_risk/total_customers):.1%}")
                    k4.metric("Monthly Revenue at Risk", f"${rev_at_risk:,.2f}")

                    # Visual Analytics
                    c_chart1, c_chart2 = st.columns(2)

                    with c_chart1:
                        # Risk Level Donut Chart
                        risk_df = pred_results["Risk Level"].value_counts().reset_index()
                        risk_df.columns = ["Risk Level", "Count"]
                        fig_donut = px.pie(
                            risk_df,
                            names="Risk Level",
                            values="Count",
                            title="Churn Risk Distribution",
                            hole=0.45,
                            color="Risk Level",
                            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)

                    with c_chart2:
                        # Probability distribution histogram
                        fig_hist = px.histogram(
                            pred_results,
                            x="Churn Probability (%)",
                            nbins=25,
                            title=f"Probability Distribution (Threshold: {decision_threshold*100:.0f}%)",
                            color="Prediction",
                            color_discrete_map={"Likely to Churn": "#ef4444", "Unlikely to Churn": "#10b981"},
                        )
                        fig_hist.add_vline(x=decision_threshold * 100, line_dash="dash", line_color="black", annotation_text="Threshold")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Interactive Filterable Results Table
                    st.markdown("#### 📋 **Enriched Prediction Results**")
                    filter_risk = st.selectbox("Filter by Risk Level:", ["All Customers", "High Risk Only", "Medium Risk Only", "Low Risk Only"])
                    display_df = pred_results.copy()
                    if filter_risk == "High Risk Only":
                        display_df = display_df[display_df["Risk Level"] == "High"]
                    elif filter_risk == "Medium Risk Only":
                        display_df = display_df[display_df["Risk Level"] == "Medium"]
                    elif filter_risk == "Low Risk Only":
                        display_df = display_df[display_df["Risk Level"] == "Low"]

                    st.dataframe(display_df, use_container_width=True)

                    # Download button
                    csv_export = pred_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Full Predictions CSV",
                        data=csv_export,
                        file_name=f"churn_predictions_threshold_{int(decision_threshold*100)}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(f"❌ Batch prediction processing failed: {exc}")


# =====================================================
# TAB 3: MODEL INSIGHTS & ARCHITECTURE
# =====================================================
with tab_insights:
    st.markdown("### 📈 **Model Development & Notebook Walkthrough**")
    st.write(
        "A synthesis of insights across the 5 project notebooks: EDA, Preprocessing, Model Training, "
        "Hyperparameter Tuning, and Threshold Validation."
    )

    st.markdown("#### 1. Notebook Pipeline Architecture")
    flow_col1, flow_col2, flow_col3 = st.columns(3)
    with flow_col1:
        st.markdown(
            """
            **Step 1: EDA & Cleaning**
            - File: `Data_Cleaning_and_EDA.ipynb`
            - Dropped `customerID`
            - Coerced blank `TotalCharges`
            - Encoded `SeniorCitizen` to string categories
            - Created `tenure_group` (1-12, 13-24, etc.)
            """
        )
    with flow_col2:
        st.markdown(
            """
            **Step 2: Preprocessing**
            - File: `Preprocessing_and_train_test_split.ipynb`
            - 80/20 Stratified train/test split
            - StandardScaler on `MonthlyCharges`, `TotalCharges`
            - OneHotEncoder on 17 categorical features
            - Produces 51 encoded features
            """
        )
    with flow_col3:
        st.markdown(
            """
            **Step 3 & 4: Training & Tuning**
            - Files: `Model_training.ipynb`, `hyperparameter_tuning.ipynb`
            - Evaluated 12 classification algorithms
            - XGBoost tuned via StratifiedKFold GridSearch
            - Optimized threshold from `0.50` to `0.30`
            """
        )

    st.divider()
    st.markdown("#### 2. Tuned Model Benchmark Comparison")
    st.write("Cross-validated test set comparison from `hyperparameter_tuning.ipynb`:")

    comp_data = {
        "Model": ["XGBoost (Tuned)", "AdaBoost", "Logistic Regression", "Gaussian Naive Bayes"],
        "Accuracy": [0.7953, 0.7953, 0.7918, 0.6894],
        "Precision": [0.6378, 0.6396, 0.6401, 0.4535],
        "Recall": [0.5321, 0.5267, 0.4947, 0.8209],
        "F1 Score": [0.5802, 0.5777, 0.5581, 0.5842],
        "ROC-AUC": [0.8378, 0.8344, 0.8306, 0.8078],
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True)

    st.divider()
    st.markdown("#### 3. Why the 0.30 Threshold?")
    st.write(
        "At the default threshold of `0.50`, the model misses nearly **47%** of churning customers. "
        "In telecom, customer acquisition cost (CAC) is 5x to 7x higher than retention cost. "
        "By optimizing the decision boundary to **0.30**, recall jumps from **53.2% to 76.5%**, "
        "enabling proactive intervention on over three-quarters of at-risk accounts."
    )

    t_metrics_col1, t_metrics_col2 = st.columns(2)
    with t_metrics_col1:
        st.metric("Recall at Default 0.50 Threshold", "53.2%", help="Captures barely half of all churners")
    with t_metrics_col2:
        st.metric("Recall at Tuned 0.30 Threshold", "76.5%", "+23.3% Lift in Churn Detection", help="Catches over 3 out of 4 churners")

    # Feature Importance Chart
    st.divider()
    st.markdown("#### 4. Top Feature Importances (Trained XGBoost)")
    fi_feature_names = preprocessor.get_feature_names_out()
    fi_df = pd.DataFrame({
        "Feature": fi_feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False).head(12)

    fig_fi = px.bar(
        fi_df.sort_values(by="Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 12 Features Driving XGBoost Predictions",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_fi.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_fi, use_container_width=True)


# =====================================================
# 8. FOOTER
# =====================================================
st.divider()
st.caption(
    f"🔮 ChurnIQ Customer Retention Intelligence System | Tuned XGBoost Engine | "
    f"Active Classification Threshold: {decision_threshold:.2f}"
)
