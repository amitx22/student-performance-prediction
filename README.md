 # 🎓 Student Performance Prediction – Machine Learning Capstone

This project predicts **student math performance** and **overall pass/fail outcome** using demographic and academic features.

It is a complete **end-to-end ML pipeline**:

- ✅ Data Cleaning  
- ✅ Exploratory Data Analysis (EDA)  
- ✅ Feature Engineering  
- ✅ ML Model Building (Regression + Classification)  
- ✅ Model Evaluation  
- ✅ Streamlit Web App Deployment  

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://student-performance-prediction-scjkdij5iepec4xq37vhmp.streamlit.app/

---

## 📊 Project Overview

The project is built on a student performance dataset that contains:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch` (standard / free-reduced)
- `test preparation course` (none / completed)
- `math score`
- `reading score`
- `writing score`

### 🎯 Project Goals

#### 1️⃣ Predict *Math Score* (Regression)

Given the student's background + reading & writing scores, the model predicts their **math score**.

Models experimented with:

- Linear Regression  
- Random Forest Regressor ✅ *(final chosen model)*  

The **Random Forest Regressor** was selected based on better generalization and performance.

---

#### 2️⃣ Predict *Pass/Fail* (Classification)

A custom rule was used to define pass/fail:

\[
\text{Average Score} = \frac{\text{Math} + \text{Reading} + \text{Writing}}{3}
\]

- If **Average ≥ 60 → Pass**
- Else → **Fail**

Models used:

- Logistic Regression  
- Random Forest Classifier ✅ *(final chosen model)*  

The **Random Forest Classifier** was chosen for its ability to capture non-linear relationships and better classification performance.

---

#### 3️⃣ Deploy ML Models as a Web App

A user-friendly **Streamlit** web application was built where users can:

- Provide **student details** (gender, race/ethnicity, etc.)
- Input **reading & writing scores**
- Get:
  - Predicted **Math score**
  - Predicted **Pass/Fail**
  - Model confidence (probability of passing)
  - Simple **study suggestions**

---

## 🧠 ML Workflow

1. **Data Understanding & Cleaning**
   - Handled missing or inconsistent values  
   - Standardized categorical labels  
   - Checked distributions & outliers  

2. **Exploratory Data Analysis (EDA)**
   - Visualized score distributions  
   - Compared performance across:
     - Gender  
     - Race/Ethnicity  
     - Lunch type  
     - Test preparation  

3. **Feature Engineering**
   - Created **Pass/Fail target** based on average score  
   - One-hot encoded categorical variables  
   - Split data into train/test sets  

4. **Model Training**
   - Trained:
     - Random Forest Regressor for `math_score`
     - Random Forest Classifier for `pass/fail`
   - Tuned basic hyperparameters  
   - Evaluated using regression & classification metrics  

5. **Model Saving**
   - Trained models saved using `joblib`:
     - `rf_reg_math.pkl`
     - `rf_clf_pass.pkl`
     - `scaler.pkl` (if any scaling used)
     - `feature_columns.pkl` (to align input features with training)

6. **Deployment**
   - Built a **Streamlit app (`app.py`)**  
   - Integrated pre-trained ML models  
   - Deployed on **Streamlit Community Cloud**

---

## 🧩 Tech Stack

**Language:**
- Python 3.x

**Libraries:**
- `pandas` – data handling  
- `numpy` – numerical operations  
- `scikit-learn` – ML models, preprocessing  
- `joblib` – model serialization  
- `matplotlib` / `seaborn` – EDA & visualization  
- `streamlit` – web app framework  

---

## 📂 Project Structure

```bash
student-performance-prediction/
│
├── app.py                   # Streamlit app
├── requirements.txt         # Project dependencies
├── runtime.txt              # Python runtime version (for deployment)
│
├── models/                  # Saved ML artifacts
│   ├── rf_reg_math.pkl
│   ├── rf_clf_pass.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/               # Jupyter notebooks (EDA, training)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
└── data/
    └── students_performance.csv   # Source dataset (not always committed)
