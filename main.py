# Employee Performance Predictor (Windows Ready)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Create folders if not exist
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. CREATE SYNTHETIC DATA
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

# Performance calculation
data["PerformanceScore"] = (
    (data["Experience"] * 0.3) +
    (data["TrainingHours"] * 0.4) +
    (data["Salary"] / 10000 * 0.3)
)

# Convert to categories
data["Performance"] = pd.cut(
    data["PerformanceScore"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

# Save dataset
data.to_csv("data/employees.csv", index=False)

print("\n✅ Dataset Created Successfully!\n")
print(data.head())

# -----------------------------
# 2. PREPROCESSING
# -----------------------------

le = LabelEncoder()

data["Department"] = le.fit_transform(data["Department"])
data["Performance"] = le.fit_transform(data["Performance"])

X = data.drop(["Performance", "PerformanceScore"], axis=1)
y = data["Performance"]

# -----------------------------
# 3. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. MODEL TRAINING
# -----------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. PREDICTION
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# 6. EVALUATION
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:\n", cm)

# -----------------------------
# 7. VISUALIZATION
# -----------------------------

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")

# Feature Importance
plt.figure()
importances = model.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")

print("\n📁 Outputs saved in 'outputs/' folder")

# -----------------------------
# 8. TEST NEW PREDICTION
# -----------------------------

sample = [[30, 5, 50000, 1, 40]]  # Age, Experience, Salary, Dept, TrainingHours
prediction = model.predict(sample)

print("\n🧠 Sample Prediction (0=Low,1=Medium,2=High):", prediction)