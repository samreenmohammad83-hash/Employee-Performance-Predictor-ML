# Streamlit Employee Performance Predictor UI

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Employee Performance Predictor", layout="centered")

st.title("📊 Employee Performance Predictor")
st.write("Predict employee performance using Machine Learning")

# -----------------------------
# CREATE SAME DATA (for training)
# -----------------------------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "Age": np.random.randint(22, 60, n),
    "Experience": np.random.randint(1, 20, n),
    "Salary": np.random.randint(20000, 120000, n),
    "Department": np.random.choice(["HR", "IT", "Sales", "Finance"], n),
    "TrainingHours": np.random.randint(10, 100, n)
})

data["PerformanceScore"] = (
    (data["Experience"] * 0.3) +
    (data["TrainingHours"] * 0.4) +
    (data["Salary"] / 10000 * 0.3)
)

data["Performance"] = pd.cut(
    data["PerformanceScore"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# ENCODING
# -----------------------------
le_dept = LabelEncoder()
data["Department"] = le_dept.fit_transform(data["Department"])

le_perf = LabelEncoder()
data["Performance"] = le_perf.fit_transform(data["Performance"])

X = data.drop(["Performance", "PerformanceScore"], axis=1)
y = data["Performance"]

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# USER INPUT UI
# -----------------------------
st.subheader("🧾 Enter Employee Details")

age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience (Years)", 0, 20, 5)
salary = st.slider("Salary", 20000, 120000, 50000)
department = st.selectbox("Department", ["HR", "IT", "Sales", "Finance"])
training = st.slider("Training Hours", 10, 100, 40)

# Encode department
dept_encoded = le_dept.transform([department])[0]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Performance"):
    
    input_data = np.array([[age, experience, salary, dept_encoded, training]])
    
    prediction = model.predict(input_data)[0]

    # Convert label back
    if prediction == 0:
        result = "Low"
        st.error(f"Performance Level: {result}")
    elif prediction == 1:
        result = "Medium"
        st.warning(f"Performance Level: {result}")
    else:
        result = "High"
        st.success(f"Performance Level: {result}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Built with ❤️ using Streamlit")