from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# Load saved objects
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    df = pd.DataFrame([data])

    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Encode categorical values
    for col, encoder in encoders.items():
        if col in df.columns:
            if df[col].iloc[0] not in encoder.classes_:
                df[col] = encoder.transform([encoder.classes_[0]])
            else:
                df[col] = encoder.transform(df[col])

    # Match training column order
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    attrition_result = target_encoder.inverse_transform([prediction])[0]

    risk_percent = round(probability * 100, 2)

    # SHAP explanation
    shap_values = explainer.shap_values(df_scaled)

    contributions = dict(zip(feature_columns, shap_values[0]))

    # Sort most influential features
    top_factors = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return render_template(
        "result.html",
        prediction=attrition_result,
        probability=risk_percent,
        employee=data,
        factors=top_factors
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
