import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_config(file_path):
    """Load configuration from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Error decoding JSON file {file_path}: {e}")

def preprocess_data(file_path, config_dir):
    try:
        # Load configuration files
        columns = load_config(f"{config_dir}/columns.json")['columns']
        training_columns = load_config(f"{config_dir}/training_columns.json")['training_columns']

        # Load dataset
        df = pd.read_csv(file_path)

        # Check column consistency
        if not set(columns) <= set(df.columns):
            raise ValueError("Dataset columns do not match the configuration file.")

        # Filter columns and handle outliers
        df = df[columns]
        for col in training_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Encode target variable
        label_encoder = LabelEncoder()
        df['Crop'] = label_encoder.fit_transform(df['Crop'])

        # Split into features and target
        X = df[training_columns]
        y = df['Crop']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, label_encoder, scaler

    except Exception as e:
        raise RuntimeError(f"An error occurred during preprocessing: {e}")
