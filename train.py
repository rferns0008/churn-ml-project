import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Generate synthetic dataset matching your API schema
np.random.seed(42)
n_samples = 1000

print("Generating synthetic dataset...")
X = pd.DataFrame({
    "tenure": np.random.randint(1, 72, n_samples),
    "monthly_charges": np.random.uniform(20.0, 120.0, n_samples),
    "total_charges": np.random.uniform(20.0, 8600.0, n_samples),
    "contract_type": np.random.choice([0, 1, 2], n_samples),   # 0: Month-to-month, 1: 1yr, 2: 2yr
    "internet_service": np.random.choice([0, 1, 2], n_samples) # 0: DSL, 1: Fiber, 2: No
})

# Create a target variable 'churn' (0 = Retained, 1 = Churned)
# We add synthetic logic so the model learns realistic patterns (higher churn for short tenure/high charges)
churn_probability = (X["monthly_charges"] / 120.0) * 0.4 + (1 - X["tenure"] / 72.0) * 0.5
y = (np.random.rand(n_samples) < churn_probability).astype(int)

# 2. Train the Scikit-Learn Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# 3. Export the model as a pickle file
joblib.dump(model, "model.pkl")
print("✅ Success! model.pkl has been saved to the current directory.")