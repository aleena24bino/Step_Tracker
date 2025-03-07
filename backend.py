from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

STATIC_FOLDER = "static"

# Directory to store uploaded CSVs
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Path for trained model
MODEL_PATH = "trained_model.pkl"

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None

load_model()
training_status = {"status": "idle"}

def train_model():
    """Train the model using the uploaded CSV file."""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "user_uploaded.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError("No dataset found. Please upload a CSV.")

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True)
        df["Day"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Week"] = df["Date"].dt.isocalendar().week

        X = df[["Day", "Month", "Year", "Week"]]
        y = df["StepCount"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)

        # Generate Weekly Trend Graph
        weekly_trend = df.groupby("Week")["StepCount"].mean()
        plt.figure(figsize=(8, 4))
        plt.plot(weekly_trend.index, weekly_trend.values, marker="o", linestyle="-", color="b")
        plt.xlabel("Week Number")
        plt.ylabel("Average Steps")
        plt.title("Weekly Step Count Trend")
        plt.grid(True)
        plt.savefig(os.path.join(STATIC_FOLDER, "weekly_trend.png"))
        plt.close()

        # Generate Monthly Trend Graph
        monthly_trend = df.groupby("Month")["StepCount"].mean()
        plt.figure(figsize=(8, 4))
        plt.plot(monthly_trend.index, monthly_trend.values, marker="o", linestyle="-", color="g")
        plt.xlabel("Month")
        plt.ylabel("Average Steps")
        plt.title("Monthly Step Count Trend")
        plt.grid(True)
        plt.savefig(os.path.join(STATIC_FOLDER, "monthly_trend.png"))
        plt.close()

    except Exception as e:
        training_status["status"] = "Training failed!"
        raise e

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    global training_status
    try:
        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, "user_uploaded.csv")
        file.save(file_path)
        training_status["status"] = "Training in progress..."
        train_model()
        training_status["status"] = "Training Completed!"
        load_model()
        return jsonify({"message": "Model training completed!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/training_status", methods=["GET"])
def get_training_status():
    return jsonify(training_status)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        global model
        if model is None:
            return jsonify({"error": "Model not trained yet!"}), 400

        data = request.json
        input_date = pd.to_datetime(data["date"])
        input_features = pd.DataFrame([{ "Day": input_date.day, "Month": input_date.month, "Year": input_date.year, "Week": input_date.isocalendar().week }])
        prediction = model.predict(input_features)[0]
        activity_status = "Active Day" if prediction >= 10000 else "Inactive Day"
        health_tips = "Great job! Keep maintaining your step count!" if activity_status == "Active Day" else "Consider taking short walks, using stairs, or setting step goals."
        return jsonify({"predicted_steps": prediction, "activity_status": activity_status, "health_tips": health_tips})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/weekly_trend")
def get_weekly_trend():
    file_path = os.path.join(STATIC_FOLDER, "weekly_trend.png")
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Weekly trend graph not found!"}), 404  # Return error if file is missing

    return send_file(file_path, mimetype="image/png")


@app.route("/monthly_trend")
def get_monthly_trend():
    file_path = os.path.join(STATIC_FOLDER, "monthly_trend.png")
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Monthly trend graph not found!"}), 404  # Return error if file is missing

    return send_file(file_path, mimetype="image/png")

@app.route("/clear_data", methods=["POST"])
def clear_data():
    global model, training_status

    try:
        # Clear all stored files and graphs
        clear_old_data()

        # Reset global variables
        model = None
        training_status = {"status": "idle"}

        return jsonify({"message": "All user data has been cleared!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



def clear_old_data():
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    for file in ["weekly_trend.png", "monthly_trend.png"]:
        path = os.path.join(STATIC_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)

clear_old_data()

if __name__ == "__main__":
    app.run(debug=True)
