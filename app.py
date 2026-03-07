from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# Load saved objects
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    df = pd.DataFrame([data])


    # Convert numeric values
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass


    # Encode categorical values
    for col, encoder in encoders.items():
        if col in df.columns:

            # Handle unseen categories
            if df[col].iloc[0] not in encoder.classes_:
                df[col] = encoder.transform([encoder.classes_[0]])
            else:
                df[col] = encoder.transform(df[col])


    # Ensure column order matches training
    df = df.reindex(columns=feature_columns, fill_value=0)


    # Scale
    df_scaled = scaler.transform(df)


    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    attrition_result = target_encoder.inverse_transform([prediction])[0]

    risk_percent = round(probability * 100, 2)


    return render_template(
        "result.html",
        prediction=attrition_result,
        probability=risk_percent,
        employee=data
    )


if __name__ == "__main__":
    app.run(debug=False)
