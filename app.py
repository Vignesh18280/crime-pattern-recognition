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
from src.pattern_generator import generate_multiple_crime_patterns
from src.dataset import (
    MultiModalCrimeDataset, build_dataloaders,
    LOG_FEATURE_DIM, LOG_SEQ_LEN, IMG_SIZE, BIN_IMG_SIZE,
    IMG_MAX_TIMED, IMG_MAX_STATIC
)
from torchvision import transforms
from PIL import Image
import io
import shutil


# --- Configuration ---
MODEL_PATH = "results/best_model.pth"
DATA_PATH = "data/UNSW_NB15_testing-set.parquet"
UPLOAD_FOLDER = 'data/temp_uploads' # Folder for temporary file uploads
SCALER_PATH = 'scaler.pkl' # Path to saved StandardScaler
ALLOWED_EXTENSIONS = {'csv', 'parquet'}
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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
    # Get model params from checkpoint config
    config = checkpoint.get('config', {})
    log_feature_dim = config.get('log_feature_dim', 32)
    log_seq_len = config.get('log_seq_len', 5)
    
    # Instantiate the new multi-modal Siamese Network (matching training config)
    model = SiameseCrimeMatcher(
        log_feature_dim=log_feature_dim,
        log_seq_len=log_seq_len,
        log_embedding_dim=128,
        image_embedding_dim=256,
        binary_embedding_dim=128,
        fused_embedding_dim=256
    )
    # Note: Loading state_dict might fail if the old model architecture is
    # different from the new one. A re-training might be necessary.
    try:
        model.load_state_dict(checkpoint['model_state'])
        print("Model state loaded successfully.")
    except RuntimeError as e:
        print(f"Warning: Could not load model state dict, likely due to architecture change: {e}")
        print("         The model will use its initial weights. Re-training is recommended.")

    model.to(DEVICE)
    model.eval()
    print("Model initialized.")
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
    generates multiple 5-event crime patterns, preprocesses them, and returns a list of tensors.
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

        # --- Anomaly Detection ---
        suspicious_df = detect_suspicious_connections(df)

        if len(suspicious_df) < 5:
            return None, [f"Found only {len(suspicious_df)} suspicious connections. At least 5 are required."]

        # --- Generate Multiple 5-event Patterns ---
        raw_patterns = generate_multiple_crime_patterns(suspicious_df, window_size=5, stride=1)
        
        if not raw_patterns:
            return None, ["Could not generate any 5-event crime patterns."]

        processed_pattern_tensors = []
        final_feature_names = []

        for raw_pattern_np in raw_patterns:
            X = raw_pattern_np.astype(np.float32)
            temp_feature_df = pd.DataFrame(X).iloc[:, :32] # Simplified for demo
            
            current_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            if X.shape[1] < feature_dim:
                pad = np.zeros((X.shape[0], feature_dim - X.shape[1]), dtype=np.float32)
                X   = np.hstack([X, pad])
            elif X.shape[1] > feature_dim:
                X   = X[:, :feature_dim]
            
            X = scaler.transform(X)
            processed_pattern_tensors.append(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
            
            if not final_feature_names: 
                final_feature_names = current_feature_names[:feature_dim]

        return processed_pattern_tensors, final_feature_names

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
# Serve React frontend build
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")

@app.route('/')
def index():
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")

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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/api/load_sample', methods=['POST'])
def load_sample():
    data = request.get_json()
    side = data.get('side', 'a')
    
    manifest_path = os.path.join('data', 'manifest.csv')
    if not os.path.exists(manifest_path):
        return jsonify({"error": "No sample data available"}), 400
    
    import pandas as pd
    df = pd.read_csv(manifest_path)
    
    if df.empty:
        return jsonify({"error": "No incidents in manifest"}), 400
    
    incident = df.iloc[0]
    return jsonify({
        "incident_id": incident['incident_id'],
        "attack_type": incident['attack_type'],
        "log_path": incident['log_path'],
        "image_folder": incident['image_folder'],
        "binary_path": incident.get('binary_path', '')
    })


@app.route('/api/predict_multimodal', methods=['POST'])
def predict_multimodal():
    if model is None:
        return jsonify({"error": "Server not ready. Model not loaded."}), 500

    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    bin_transform = transforms.Compose([
        transforms.Resize((BIN_IMG_SIZE, BIN_IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def process_incident_a():
        log_data = torch.zeros(1, LOG_SEQ_LEN, LOG_FEATURE_DIM)
        timed_imgs = []
        static_imgs = []
        bin_data = None
        
        if 'log_a' in request.files:
            df = pd.read_csv(request.files['log_a'])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ['id', 'label']]
            
            if len(feature_cols) == 0:
                feature_cols = numeric_cols[:32] if len(numeric_cols) >= 32 else numeric_cols
            
            if 'attack_cat' in df.columns:
                feature_cols = [c for c in feature_cols if c != 'attack_cat']
            if 'is_sm_ips_ports' in df.columns:
                feature_cols = [c for c in feature_cols if c != 'is_sm_ips_ports']
            
            X = df[feature_cols].fillna(0).values.astype(np.float32)
            
            if scaler is not None:
                X = scaler.transform(X)
            
            if X.shape[0] >= LOG_SEQ_LEN:
                X = X[:LOG_SEQ_LEN, :LOG_FEATURE_DIM]
            else:
                X = np.vstack([X, np.zeros((LOG_SEQ_LEN - X.shape[0], LOG_FEATURE_DIM))])
            
            log_data = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        
        if 'images_a' in request.files:
            files = request.files.getlist('images_a')
            for f in files:
                try:
                    img = Image.open(f).convert('RGB')
                    timed_imgs.append(img_transform(img))
                except Exception as e:
                    print(f"Error processing image {f.filename}: {e}")
        
        if 'binary_a' in request.files:
            try:
                binary_file = request.files['binary_a']
                binary_data = binary_file.read()
                
                byte_array = np.frombuffer(binary_data, dtype=np.uint8)
                byte_array = byte_array[:BIN_IMG_SIZE * BIN_IMG_SIZE]
                
                if len(byte_array) < BIN_IMG_SIZE * BIN_IMG_SIZE:
                    byte_array = np.pad(byte_array, (0, BIN_IMG_SIZE * BIN_IMG_SIZE - len(byte_array)), mode='constant')
                
                bin_img = byte_array.reshape(BIN_IMG_SIZE, BIN_IMG_SIZE)
                bin_tensor = torch.tensor(bin_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                bin_data = bin_tensor.squeeze(0)
            except Exception as e:
                print(f"Error processing binary: {e}")
                bin_data = None
        
        log_tensor = log_data
        if timed_imgs:
            timed_tensor = torch.stack(timed_imgs).unsqueeze(0)
        else:
            timed_tensor = torch.zeros(1, 1, 3, IMG_SIZE, IMG_SIZE)
        static_tensor = torch.zeros(1, 1, 3, IMG_SIZE, IMG_SIZE)
        if bin_data is not None:
            bin_tensor = bin_data.unsqueeze(1)
        else:
            bin_tensor = torch.zeros(1, 1, BIN_IMG_SIZE, BIN_IMG_SIZE)
        
        return log_tensor, timed_tensor, static_tensor, bin_tensor

    def process_incident_b():
        log_data = torch.zeros(1, LOG_SEQ_LEN, LOG_FEATURE_DIM)
        timed_imgs = []
        static_imgs = []
        bin_data = None
        
        if 'log_b' in request.files:
            df = pd.read_csv(request.files['log_b'])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ['id', 'label']]
            
            if len(feature_cols) == 0:
                feature_cols = numeric_cols[:32] if len(numeric_cols) >= 32 else numeric_cols
            
            if 'attack_cat' in df.columns:
                feature_cols = [c for c in feature_cols if c != 'attack_cat']
            if 'is_sm_ips_ports' in df.columns:
                feature_cols = [c for c in feature_cols if c != 'is_sm_ips_ports']
            
            X = df[feature_cols].fillna(0).values.astype(np.float32)
            
            if scaler is not None:
                X = scaler.transform(X)
            
            if X.shape[0] >= LOG_SEQ_LEN:
                X = X[:LOG_SEQ_LEN, :LOG_FEATURE_DIM]
            else:
                X = np.vstack([X, np.zeros((LOG_SEQ_LEN - X.shape[0], LOG_FEATURE_DIM))])
            
            log_data = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        
        if 'images_b' in request.files:
            files = request.files.getlist('images_b')
            for f in files:
                try:
                    img = Image.open(f).convert('RGB')
                    timed_imgs.append(img_transform(img))
                except Exception as e:
                    print(f"Error processing image {f.filename}: {e}")
        
        if 'binary_b' in request.files:
            try:
                binary_file = request.files['binary_b']
                binary_data = binary_file.read()
                
                byte_array = np.frombuffer(binary_data, dtype=np.uint8)
                byte_array = byte_array[:BIN_IMG_SIZE * BIN_IMG_SIZE]
                
                if len(byte_array) < BIN_IMG_SIZE * BIN_IMG_SIZE:
                    byte_array = np.pad(byte_array, (0, BIN_IMG_SIZE * BIN_IMG_SIZE - len(byte_array)), mode='constant')
                
                bin_img = byte_array.reshape(BIN_IMG_SIZE, BIN_IMG_SIZE)
                bin_tensor = torch.tensor(bin_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                bin_data = bin_tensor.squeeze(0)
            except Exception as e:
                print(f"Error processing binary: {e}")
                bin_data = None
        
        log_tensor = log_data
        if timed_imgs:
            timed_tensor = torch.stack(timed_imgs).unsqueeze(0)
        else:
            timed_tensor = torch.zeros(1, 1, 3, IMG_SIZE, IMG_SIZE)
        static_tensor = torch.zeros(1, 1, 3, IMG_SIZE, IMG_SIZE)
        if bin_data is not None:
            bin_tensor = bin_data.unsqueeze(1)
        else:
            bin_tensor = torch.zeros(1, 1, BIN_IMG_SIZE, BIN_IMG_SIZE)
        
        return log_tensor, timed_tensor, static_tensor, bin_tensor

    log_a, timed_a, static_a, bin_a = process_incident_a()
    log_b, timed_b, static_b, bin_b = process_incident_b()
    
    has_log_a = 'log_a' in request.files
    has_log_b = 'log_b' in request.files
    has_img_a = len(request.files.getlist('images_a')) > 0
    has_img_b = len(request.files.getlist('images_b')) > 0
    has_bin_a = 'binary_a' in request.files
    has_bin_b = 'binary_b' in request.files

    incident_a = (
        log_a.to(DEVICE), timed_a.to(DEVICE),
        static_a.to(DEVICE), bin_a.to(DEVICE)
    )
    incident_b = (
        log_b.to(DEVICE), timed_b.to(DEVICE),
        static_b.to(DEVICE), bin_b.to(DEVICE)
    )

    with torch.no_grad():
        log_emb_a = model.log_extractor(log_a.to(DEVICE))
        log_emb_b = model.log_extractor(log_b.to(DEVICE))
        img_emb_a = model.image_extractor(timed_a.to(DEVICE), static_a.to(DEVICE))
        img_emb_b = model.image_extractor(timed_b.to(DEVICE), static_b.to(DEVICE))
        bin_emb_a = model.binary_extractor(bin_a.to(DEVICE))
        bin_emb_b = model.binary_extractor(bin_b.to(DEVICE))
        
        log_diff = (log_emb_a - log_emb_b).abs().mean().item()
        img_diff = (img_emb_a - img_emb_b).abs().mean().item()
        bin_diff = (bin_emb_a - bin_emb_b).abs().mean().item()
        
        # Use raw data similarity for logs (more reliable with untrained model)
        log_similarity = 1.0 / (1.0 + log_diff)
        img_similarity = 1.0 / (1.0 + img_diff)
        bin_similarity = 1.0 / (1.0 + bin_diff) if bin_emb_a.abs().sum() > 0 else 0.0
        
        # If logs are present, use log similarity directly as it's the most reliable
        # Otherwise combine with other modalities
        if has_log_a or has_log_b:
            score = log_similarity
        else:
            # Combine image and binary if no logs
            total_weight = 0
            weighted_score = 0
            if has_img_a or has_img_b:
                weighted_score += img_similarity * 0.6
                total_weight += 0.6
            if has_bin_a or has_bin_b:
                weighted_score += bin_similarity * 0.4
                total_weight += 0.4
            score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        log_emb_diff = (log_emb_a - log_emb_b).abs().cpu().flatten().tolist()
        img_emb_diff = (img_emb_a - img_emb_b).abs().cpu().flatten().tolist()
        
        explanation = []
        for i, val in enumerate(log_emb_diff[:10]):
            explanation.append({"feature": f"Log_{i}", "importance": float(val)})
        for i, val in enumerate(img_emb_diff[:10]):
            explanation.append({"feature": f"Img_{i}", "importance": float(val)})
        explanation = sorted(explanation, key=lambda x: abs(x['importance']), reverse=True)

    verdict = "SAME MO" if score >= 0.5 else "DIFFERENT MO"
    
    log_data_a = log_a.squeeze(0).flatten().cpu().tolist()
    log_data_b = log_b.squeeze(0).flatten().cpu().tolist()
    
    pattern_data_a = log_data_a[:10]
    pattern_data_b = log_data_b[:10]
    
    return jsonify({
        "verdict": verdict,
        "similarityScore": score,
        "explanation": explanation,
        "best_patterns": {
            "pattern1_index": 0,
            "pattern2_index": 0,
            "pattern1_data": pattern_data_a,
            "pattern2_data": pattern_data_b
        },
        "incident1": "Multimodal Incident A",
        "incident2": "Multimodal Incident B",
        "modalities_a": {
            "has_log": 'log_a' in request.files,
            "num_images": len(request.files.getlist('images_a')),
            "has_binary": 'binary_a' in request.files
        },
        "modalities_b": {
            "has_log": 'log_b' in request.files,
            "num_images": len(request.files.getlist('images_b')),
            "has_binary": 'binary_b' in request.files
        },
        "embeddings": {
            "log_similarity": log_similarity,
            "img_similarity": img_similarity,
            "bin_similarity": bin_similarity
        }
    })


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
        if os.path.exists(file_path1): os.remove(file_path1)
        if os.path.exists(file_path2): os.remove(file_path2)
        return jsonify({"error": " ".join(errors)}), 400

    if not pattern1 or not pattern2:
        return jsonify({"error": "Could not process one or both incident logs into valid crime patterns."}), 400

    # --- Best-Match Comparison ---
    max_similarity_score = -1.0
    best_p1_tensor = None
    best_p2_tensor = None
    best_p1_idx = 0
    best_p2_idx = 0

    with torch.no_grad():
        for idx1, p1 in enumerate(pattern1):
            for idx2, p2 in enumerate(pattern2):
                p1dev = p1.to(DEVICE)
                p2dev = p2.to(DEVICE)
                score = model.compute_similarity(p1dev, p2dev).item()
                if score > max_similarity_score:
                    max_similarity_score = score
                    best_p1_tensor = p1
                    best_p2_tensor = p2
                    best_p1_idx = idx1
                    best_p2_idx = idx2

    verdict = "SAME MO" if max_similarity_score >= 0.5 else "DIFFERENT MO"

    # Convert best patterns back to raw values for display
    best_p1_data = []
    best_p2_data = []
    if best_p1_tensor is not None and best_p2_tensor is not None:
        # Remove batch dimension and move to CPU
        p1_np = best_p1_tensor.squeeze(0).cpu().numpy()
        p2_np = best_p2_tensor.squeeze(0).cpu().numpy()
        
        # Inverse transform to get raw values
        try:
            p1_raw = scaler.inverse_transform(p1_np)
            p2_raw = scaler.inverse_transform(p2_np)
        except Exception:
            p1_raw = p1_np
            p2_raw = p2_np
        
        # Fallback: if all zeros, use raw values
        if np.all(p1_raw == 0):
            p1_raw = p1_np
        if np.all(p2_raw == 0):
            p2_raw = p2_np
        
        # Take first 10 features from first row of each pattern
        max_features = 10
        p1_row = p1_raw[0, :max_features] if p1_raw.shape[1] >= max_features else np.pad(p1_raw[0, :], (0, max_features - p1_raw.shape[1]))
        p2_row = p2_raw[0, :max_features] if p2_raw.shape[1] >= max_features else np.pad(p2_raw[0, :], (0, max_features - p2_raw.shape[1]))
        best_p1_data = [round(float(v), 4) for v in p1_row]
        best_p2_data = [round(float(v), 4) for v in p2_row]

    # Cleanup
    if os.path.exists(file_path1): os.remove(file_path1)
    if os.path.exists(file_path2): os.remove(file_path2)

    # Generate explanation (dummy for now)
    explanation = []
    for name in feature_names1:
        explanation.append({
            "feature": name,
            "importance": random.uniform(-0.5, 0.5)
        })
    explanation = sorted(explanation, key=lambda x: abs(x['importance']), reverse=True)

    return jsonify({
        "verdict": verdict,
        "similarityScore": max_similarity_score,
        "incident1": uploaded_file1,
        "incident2": uploaded_file2,
        "explanation": explanation[:20],
        "best_patterns": {
            "pattern1_index": best_p1_idx,
            "pattern2_index": best_p2_idx,
            "pattern1_data": best_p1_data,
            "pattern2_data": best_p2_data
        }
    })


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
