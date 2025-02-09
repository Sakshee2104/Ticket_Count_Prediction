import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import xgboost as xgb

app = Flask(__name__)

# Load trained XGBoost model
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Encoding logic for categories
CATEGORY_ENCODING = {
    "IT": 0,
    "HR": 1,
    "Software": 2,
    "Finance": 3,
    "Admin": 4
}

# Expected feature order (must match training)
FEATURE_COLUMNS = ["Year", "Month", "Day", "Hour", "DayOfWeek", "CategoryEncoded"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        timestamp = request.form["timestamp"]
        category = request.form["category"]

        # Convert timestamp into useful features
        timestamp_features = pd.to_datetime(timestamp)
        year = timestamp_features.year
        month = timestamp_features.month
        day = timestamp_features.day
        hour = timestamp_features.hour
        day_of_week = timestamp_features.dayofweek  # Monday=0, Sunday=6

        # Encode category
        if category not in CATEGORY_ENCODING:
            return render_template(
                "index.html",
                prediction="Error: Invalid category selected."
            )
        category_encoded = CATEGORY_ENCODING[category]

        # Combine features into an array
        input_data = pd.DataFrame(
            [[year, month, day, hour, day_of_week, category_encoded]],
            columns=FEATURE_COLUMNS
        )

        # Convert the input data into a DMatrix
        dmatrix_data = xgb.DMatrix(input_data)

        # Predict using the XGBoost model
        prediction = model.predict(dmatrix_data)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Ticket Count: {prediction:.2f}",
        )
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
