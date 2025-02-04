from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("car_price_model.pkl")


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from request
    features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
    prediction = model.predict(features)[0]  # Make prediction
    return jsonify({"predicted_price": prediction})

if __name__ == '__main__':
    app.run(debug=True)
