
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "tenure": np.random.randint(1, 73, n),
    "monthly_charges": np.random.randint(20, 121, n),
    "support_calls": np.random.randint(0, 11, n)
})

risk = (
    (df["tenure"] < 12).astype(int)
    + (df["support_calls"] >= 5).astype(int)
    + (df["monthly_charges"] > 80).astype(int)
)

df["churn"] = (risk >= 2).astype(int)

X = df[["tenure", "monthly_charges", "support_calls"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))

joblib.dump(model, "churn_model.pkl")
df.to_csv("churn_data.csv", index=False)

print("SUCCESS: Model saved as churn_model.pkl")
print("SUCCESS: Dataset saved as churn_data.csv")
