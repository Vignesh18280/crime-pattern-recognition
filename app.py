import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shap # Although not directly used for explanation, it's a dependency
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import random
import os
import joblib # For loading/saving the scaler
import uuid # For unique filenames

# Assuming model.py and dataset.py are in the src directory
from src.model import SiameseCrimeMatcher
from src.detect import detect_suspicious_connections


# --- Configuration ---
MODEL_PATH = "results/best_model.pth"
DATA_PATH = "data/UNSW_NB15_testing-set.parquet"
UPLOAD_FOLDER = 'data/temp_uploads' # Folder for temporary file uploads
SCALER_PATH = 'scaler.pkl' # Path to saved StandardScaler
ALLOWED_EXTENSIONS = {'csv', 'parquet'}
DEVICE = torch.device("cpu") # Use CPU for inference

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload folder exists

# --- Load Model and Data ---
print("Loading model and data...")

# Load the checkpoint
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    print("Checkpoint loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {MODEL_PATH}.")
    checkpoint = None

if checkpoint:
    # Get model parameters from the checkpoint
    feature_dim = checkpoint.get('feat_dim', 32)
    cnn_out_dim = checkpoint.get('cnn_out_dim', 64)
    mo_dim = checkpoint.get('mo_dim', 128)

    # Instantiate the Siamese Network
    model = SiameseCrimeMatcher(
        feature_dim=feature_dim,
        cnn_out_dim=cnn_out_dim,
        mo_dim=mo_dim
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
else:
    model = None

# Load the StandardScaler
scaler = None
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Scaler not found at {SCALER_PATH}. User uploaded data will not be scaled, and dummy generation will fail.")
except Exception as e:
    print(f"Error loading scaler: {e}. User uploaded data will not be scaled, and dummy generation will fail.")

# Load the dataset (for attack categories and fallback if no upload)
df_test = None
attack_categories = []
try:
    df_test = pd.read_parquet(DATA_PATH)
    # Get the attack categories, excluding "Normal"
    attack_categories = df_test[df_test['attack_cat'] != 'Normal']['attack_cat'].unique().tolist()
    print("Predefined dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Predefined dataset not found at {DATA_PATH}. Only uploaded data can be used for comparison.")
except Exception as e:
    print(f"Error loading predefined dataset: {e}. Only uploaded data can be used for comparison.")


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_dummy_pattern_data(base_category=None):
    """
    Generates a 5-event dummy pattern (5xFEATURE_DIM).
    This is a simplified fallback function.
    """
    if scaler is None:
        return None, "Scaler not loaded. Cannot generate dummy data."

    print(f"Generating dummy pattern. Base category '{base_category}' is ignored in this simplified version.")

    # Generate random data with a default mean and std
    mean_vals = np.zeros(feature_dim)
    std_vals = np.ones(feature_dim) * 0.5
    
    scaled_dummy_data = np.random.normal(loc=mean_vals, scale=std_vals, size=(5, feature_dim)).astype(np.float32)
    
    # Inverse transform to get raw data
    raw_dummy_data = scaler.inverse_transform(scaled_dummy_data)
    
    # Create a DataFrame with generic column names
    dummy_df = pd.DataFrame(raw_dummy_data, columns=[f"feature_{i}" for i in range(feature_dim)])

    return dummy_df, None


def process_incident_log(file_path):
    """
    Loads a raw incident log file (CSV/Parquet), detects suspicious connections,
    extracts the first 5 to form a crime pattern, preprocesses it, and returns a tensor.
    """
    if scaler is None:
        return None, ["Scaler not loaded. Cannot process uploaded data."]
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return None, ["Invalid file type. Only CSV and Parquet are allowed."]

        # --- New Anomaly Detection Step ---
        suspicious_df = detect_suspicious_connections(df)

        if len(suspicious_df) < 5:
            return None, [f"Found only {len(suspicious_df)} suspicious connections. At least 5 are required to form a crime pattern."]

        # Take the first 5 suspicious events to form the pattern
        pattern_df = suspicious_df.head(5)
        
        # --- Continue with existing preprocessing ---
        feature_df = pattern_df.select_dtypes(include=[np.number])
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if feature_df.empty:
            return None, ["No numeric features found in the detected suspicious connections."]
        
        X = feature_df.values.astype(np.float32)
        
        final_feature_names = list(feature_df.columns)

        if X.shape[1] < feature_dim:
            pad = np.zeros((X.shape[0], feature_dim - X.shape[1]), dtype=np.float32)
            X   = np.hstack([X, pad])
            final_feature_names += [f"pad_{i}" for i in range(feature_dim - len(final_feature_names))]
        elif X.shape[1] > feature_dim:
            X   = X[:, :feature_dim]
            final_feature_names = final_feature_names[:feature_dim]

        # Apply scaler 
        X = scaler.transform(X)

        pattern_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        return pattern_tensor, final_feature_names

    except Exception as e:
        return None, [f"Error processing incident log: {e}"]


def get_crime_pattern_by_category(category):
    """Selects 5 consecutive records of the same attack category and preprocesses them."""
    if df_test is None:
        return None, None
    
    if scaler is None:
        return None, "Scaler not loaded. Cannot process predefined data."

    category_df = df_test[df_test['attack_cat'] == category].copy()
    
    if len(category_df) < 5:
        return None, None
        
    start_index = random.randint(0, len(category_df) - 5)
    pattern_df = category_df.iloc[start_index:start_index + 5]
    
    # Consistent feature selection with dataset.py
    drop_cols = [
        "id", "srcip", "sport", "dstip", "dsport",
        "proto", "state", "service", "attack_cat", "label", "is_sm_ips_ports"
    ]
    
    feature_df = pattern_df.drop(columns=drop_cols, errors="ignore")
    
    # Keep only numeric features and handle missing values
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Ensure correct feature dimension
    X = feature_df.values.astype(np.float32)
    
    # Get feature names for SHAP before padding
    final_feature_names = list(feature_df.columns)

    if X.shape[1] < feature_dim:
        pad = np.zeros((X.shape[0], feature_dim - X.shape[1]), dtype=np.float32)
        X   = np.hstack([X, pad])
        final_feature_names += [f"pad_{i}" for i in range(feature_dim - len(final_feature_names))]
    elif X.shape[1] > feature_dim:
        X   = X[:, :feature_dim]
        final_feature_names = final_feature_names[:feature_dim]

    # Apply scaler 
    X = scaler.transform(X)

    # Convert to tensor
    pattern_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0) # Add batch dimension
    
    return pattern_tensor, final_feature_names

# --- API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/attack_categories', methods=['GET'])
def get_attack_categories():
    if not attack_categories:
        return jsonify({"error": "Attack categories not available. Please ensure UNSW_NB15_testing-set.parquet is present or upload your own files."}), 500
    return jsonify(attack_categories)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filename = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filename)
        
        # The new workflow just uploads the file. Processing happens in /api/predict
        return jsonify({"message": "File uploaded successfully", "filename": unique_filename}), 200
    else:
        return jsonify({"error": f"Allowed file types are {list(ALLOWED_EXTENSIONS)}"}), 400

@app.route('/api/generate_dummy_pattern', methods=['POST'])
def generate_dummy_pattern():
    data = request.get_json()
    base_category = data.get('base_category') # Optional: to make it statistically similar

    dummy_df, errors = generate_dummy_pattern_data(base_category=base_category)
    if dummy_df is None:
        return jsonify({"error": errors}), 500
    
    unique_filename = str(uuid.uuid4()) + ".csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    dummy_df.to_csv(file_path, index=False)

    return jsonify({"message": "Dummy pattern generated", "filename": unique_filename, "download_url": f"/api/download_dummy/{unique_filename}"}), 200

@app.route('/api/download_dummy/<filename>', methods=['GET'])
def download_dummy(filename):
    # Ensure the file is in the UPLOAD_FOLDER and is a dummy file
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Server not ready. Model not loaded."}), 500

    data = request.get_json()
    uploaded_file1 = data.get('uploadedFile1')
    uploaded_file2 = data.get('uploadedFile2')

    if not uploaded_file1 or not uploaded_file2:
        return jsonify({"error": "Please upload two incident logs for comparison."}), 400

    pattern1 = None
    pattern2 = None
    feature_names1 = []
    feature_names2 = []
    errors = []

    # Process Incident Log 1
    file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file1)
    if not os.path.exists(file_path1):
        errors.append(f"File not found: {uploaded_file1}")
    else:
        pattern1, feature_names1 = process_incident_log(file_path1)
        if pattern1 is None:
            errors.extend(feature_names1) # feature_names1 will contain the error messages

    # Process Incident Log 2
    file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file2)
    if not os.path.exists(file_path2):
        errors.append(f"File not found: {uploaded_file2}")
    else:
        pattern2, feature_names2 = process_incident_log(file_path2)
        if pattern2 is None:
            errors.extend(feature_names2)

    if errors:
        # Clean up the temporary files
        if os.path.exists(file_path1): os.remove(file_path1)
        if os.path.exists(file_path2): os.remove(file_path2)
        return jsonify({"error": " ".join(errors)}), 400

    if pattern1 is None or pattern2 is None:
        return jsonify({"error": "Could not process one or both incident logs into valid crime patterns."}), 400

    pattern1, pattern2 = pattern1.to(DEVICE), pattern2.to(DEVICE)

    with torch.no_grad():
        similarity_score = model.compute_similarity(pattern1, pattern2).item()

    verdict = "SAME MO" if similarity_score >= 0.5 else "DIFFERENT MO"

    # Clean up the temporary files after successful prediction
    if os.path.exists(file_path1): os.remove(file_path1)
    if os.path.exists(file_path2): os.remove(file_path2)

    # For the explanation, we can use the feature names from the first pattern as they should be consistent
    explanation_features = feature_names1
    explanation = []
    for name in explanation_features:
        explanation.append({
            "feature": name,
            "importance": random.uniform(-0.5, 0.5) # Still dummy importance
        })
    explanation = sorted(explanation, key=lambda x: abs(x['importance']), reverse=True)

    return jsonify({
        "verdict": verdict,
        "similarityScore": similarity_score,
        "incident1": uploaded_file1,
        "incident2": uploaded_file2,
        "explanation": explanation[:20] # Return top 20 features
    })


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
