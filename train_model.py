import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("dataset/traindataset1.csv")


# Remove useless columns if they exist
df = df.drop(columns=[
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
], errors="ignore")


# Encode target
target_encoder = LabelEncoder()
df["Attrition"] = target_encoder.fit_transform(df["Attrition"])


# Encode categorical columns
label_encoders = {}

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    if col != "Attrition":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le


# Split features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

feature_columns = X.columns


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale numeric features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# XGBoost model
model = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train, y_train)


# Predictions
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# Cross validation
scores = cross_val_score(model, scaler.transform(X), y, cv=5)

print("Cross Validation Accuracy:", scores.mean())


# Save everything needed
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("Model and preprocessing objects saved successfully.")
