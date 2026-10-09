import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load data directly into X and y to bypass the Bunch attribute warning
X, y = load_iris(return_X_y=True)

# 2. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a basic classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Evaluate and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Save the trained model
joblib.dump(model, "iris_model.pkl")
print("✅ Model saved successfully as 'iris_model.pkl'.")
