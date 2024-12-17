import joblib
import json
from flask import Flask, request, jsonify
import numpy as np

# Flask app
app = Flask(__name__)

# Load configurations
CONFIG_DIR = "config"
MODEL_PATH = "model/best_rf_model.joblib"
TRAINING_COLUMNS_PATH = f"{CONFIG_DIR}/training_columns.json"

with open(TRAINING_COLUMNS_PATH, 'r') as file:
    training_columns = json.load(file)['training_columns']

# Load model
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input features
        data = request.get_json()
        features = np.array([data['features']])  # Features should match training columns

        if len(features[0]) != len(training_columns):
            raise ValueError("Input features do not match the training columns.")

        # Predict
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except KeyError:
        return jsonify({'error': 'Invalid input format. Provide features key.'}), 400
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
