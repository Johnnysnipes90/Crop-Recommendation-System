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
LABEL_MAPPING_PATH = f"{CONFIG_DIR}/label_mapping.json"

with open(TRAINING_COLUMNS_PATH, 'r') as file:
    training_columns = json.load(file)['training_columns']

with open(LABEL_MAPPING_PATH, 'r') as file:
    label_mapping = json.load(file)

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
        prediction = model.predict(features)[0]  # Get numeric prediction
        crop_name = label_mapping[str(prediction)]  # Map to crop name
        return jsonify({'prediction': crop_name})
    except KeyError:
        return jsonify({'error': 'Invalid input format. Provide features key.'}), 400
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
