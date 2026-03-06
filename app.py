from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("salary.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    experience = float(request.form["experience"])
    test = float(request.form["test_score"])
    interview = float(request.form["interview_score"])
    prediction = model.predict([[experience, test, interview]])
    return render_template("index.html", prediction_text=f"Salary: {prediction[0]}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)