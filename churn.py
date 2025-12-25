import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Drop ID
data.drop(columns=["customerid"], inplace=True)

# Convert total charges
data["total charges"] = pd.to_numeric(data["total charges"], errors="coerce")
data["total charges"].fillna(data["total charges"].mean(), inplace=True)

# Rename target
data.rename(columns={"churn value": "churn"}, inplace=True)

# Binary mapping
binary_cols = ["partner", "dependents", "phone service", "paperless billing"]
for col in binary_cols:
    data[col] = data[col].map({"Yes": 1, "No": 0})

# Drop leakage & geo text
drop_cols = [
    "count", "churn label", "churn score", "churn reason",
    "lat long", "city", "state", "zip code", "country"
]
data.drop(columns=drop_cols, inplace=True)

# One-hot encode
data = pd.get_dummies(data, drop_first=True)

# Split
X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Model & feature columns saved successfully")
