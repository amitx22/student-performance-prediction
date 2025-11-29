import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==============================
# Load Saved Models and Metadata
# ==============================

MODEL_DIR = "models"

rf_reg = joblib.load(os.path.join(MODEL_DIR, "rf_reg_math.pkl"))
rf_clf = joblib.load(os.path.join(MODEL_DIR, "rf_clf_pass.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))


# ==============================
# Streamlit App Configuration
# ==============================

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write(
    """
    Ye app tumhare CSV wale **Students Performance** data ke upar trained models ka use karke:
    
    - **Math score predict** karta hai  
    - Overall **Pass / Fail** (avg score based) ka prediction deta hai  
    """
)

st.markdown("---")

st.sidebar.header("📝 Student Details Input")

# ==============================
# Sidebar Inputs (Categorical)
# ==============================

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race_ethnicity = st.sidebar.selectbox("Race / Ethnicity", [
    "group A", "group B", "group C", "group D", "group E"
])

parent_education = st.sidebar.selectbox("Parental Level of Education", [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
])

lunch = st.sidebar.selectbox("Lunch Type", [
    "standard",
    "free/reduced"
])

test_prep = st.sidebar.selectbox("Test Preparation Course", [
    "none",
    "completed"
])

# ==============================
# Numeric Inputs (Scores)
# ==============================

st.sidebar.header("📊 Scores")

reading_score = st.sidebar.slider("Reading Score", min_value=0, max_value=100, value=70)
writing_score = st.sidebar.slider("Writing Score", min_value=0, max_value=100, value=70)

st.markdown("### 🔧 Input Summary")

st.write(
    f"""
    - **Gender:** {gender}  
    - **Race / Ethnicity:** {race_ethnicity}  
    - **Parent Education:** {parent_education}  
    - **Lunch:** {lunch}  
    - **Test Prep:** {test_prep}  
    - **Reading Score:** {reading_score}  
    - **Writing Score:** {writing_score}  
    """
)

# ==============================
# Build Input DataFrame
# ==============================

input_dict = {
    "gender": gender,
    "race_ethnicity": race_ethnicity,
    "parent_education": parent_education,
    "lunch": lunch,
    "test_prep": test_prep,
    "reading_score": reading_score,
    "writing_score": writing_score
}

input_df = pd.DataFrame([input_dict])

st.markdown("#### 🔍 Raw Input Data")
st.dataframe(input_df)

# ==============================
# Encoding: Match Training Pipeline
# ==============================

# One-hot encode same categorical columns as training
cat_cols = ["gender", "race_ethnicity", "parent_education", "lunch", "test_prep"]
input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# Add missing columns (set 0) to match training feature_cols
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Extra columns drop (if any)
input_encoded = input_encoded[feature_cols]

st.markdown("#### 🧮 Encoded Features (Model Input)")
st.dataframe(input_encoded)

# ==============================
# Prediction
# ==============================

if st.button("🚀 Predict Performance"):
    # Random Forest regressor/classifier use kar rahe hain (no scaling needed)
    X_input = input_encoded.values

    # 1) Predict Math Score
    math_pred = rf_reg.predict(X_input)[0]

    # 2) Predict Pass/Fail (overall)
    pass_pred = rf_clf.predict(X_input)[0]
    pass_prob = rf_clf.predict_proba(X_input)[0][1]  # probability of passing

    # 3) Approx avg score estimate (just to show something meaningful)
    approx_avg = (math_pred + reading_score + writing_score) / 3.0
    approx_pass_label = "Pass ✅" if approx_avg >= 60 else "Fail ❌"

    st.markdown("## 📊 Prediction Results")

    st.write(f"**Predicted Math Score:** `{math_pred:.2f}` / 100")
    st.write(f"**Random Forest - Pass Prediction:** {'Pass ✅' if pass_pred == 1 else 'Fail ❌'}")
    st.write(f"**Model Confidence (Pass Probability):** `{pass_prob*100:.1f}%`")

    st.markdown("---")
    st.markdown("### 🎯 Approx Overall Performance (Using Predicted Math + Given Reading/Writing)")

    st.write(f"- **Approx Average Score:** `{approx_avg:.2f}` / 100")
    st.write(f"- **Heuristic Pass/Fail (avg ≥ 60):** {approx_pass_label}")

    # Simple suggestions
    st.markdown("### 💡 Suggestions")
    tips = []
    if math_pred < 60:
        tips.append("- Math score low lag raha hai, basic concepts + practice questions pe focus karo.")
    if reading_score < 60:
        tips.append("- Reading score thoda kam hai, comprehension aur practice passages solve karo.")
    if writing_score < 60:
        tips.append("- Writing score improve karne ke liye essay/answer-writing daily practice karo.")
    if test_prep == "none":
        tips.append("- Test preparation course se improvement ho sakta hai, serious exam ki taiyaari ke liye try karo.")
    if lunch == "free/reduced":
        tips.append("- Health aur nutrition ka dhyaan rakhna bhi indirectly performance ko affect karta hai.")

    if not tips:
        st.success("Overall performance kaafi strong lag raha hai! 🎉 Consistency maintain karo 💪")
    else:
        for t in tips:
            st.write(t)
