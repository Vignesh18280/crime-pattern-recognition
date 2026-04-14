import os
import pandas as pd
import numpy as np
from PIL import Image

def generate_dummy_log(filepath, rows=10):
    """Generates a simple dummy CSV log file."""
    data = np.random.rand(rows, 31)
    df = pd.DataFrame(data, columns=[f'feat_{i}' for i in range(31)])
    df.to_csv(filepath, index=False)

def generate_dummy_image(filepath, size=(224, 224), add_timestamp=False):
    """Generates a random color image."""
    img_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    if add_timestamp:
        # Add minimal EXIF data with a timestamp
        from PIL.ExifTags import TAGS
        # Look up the integer tag ID for 'DateTimeOriginal'
        datetime_original_tag_id = None
        for tag_id, tag_name in TAGS.items():
            if tag_name == 'DateTimeOriginal':
                datetime_original_tag_id = tag_id
                break
        
        if datetime_original_tag_id:
            exif_dict = {datetime_original_tag_id: '2026:04:14 12:00:00'}
            exif_bytes = img.getexif()
            for tag, value in exif_dict.items():
                exif_bytes[tag] = value
            img.save(filepath, exif=exif_bytes)
        else:
            img.save(filepath) # Save without EXIF if tag not found
    else:
        img.save(filepath)

def generate_dummy_binary(filepath, size_kb=10):
    """Generates a dummy binary file."""
    with open(filepath, 'wb') as f:
        f.write(os.urandom(size_kb * 1024))

def main():
    """Main function to generate the dataset structure and manifest."""
    print("--- Generating Multi-Modal Dummy Dataset ---")

    base_dir = "data"
    incidents = {
        'incident_A': {'attack_type': 'dos', 'timed_imgs': 3, 'static_imgs': 2, 'has_binary': True},
        'incident_B': {'attack_type': 'dos', 'timed_imgs': 4, 'static_imgs': 1, 'has_binary': True},
        'incident_C': {'attack_type': 'fuzzers', 'timed_imgs': 2, 'static_imgs': 3, 'has_binary': False},
        'incident_D': {'attack_type': 'fuzzers', 'timed_imgs': 0, 'static_imgs': 5, 'has_binary': True},
        'incident_E': {'attack_type': 'backdoor', 'timed_imgs': 5, 'static_imgs': 0, 'has_binary': False},
    }
    
    manifest_data = []

    for incident_id, config in incidents.items():
        print(f"Generating data for {incident_id}...")
        
        # Create directories
        log_dir = os.path.join(base_dir, 'logs')
        img_dir = os.path.join(base_dir, 'images', incident_id)
        bin_dir = os.path.join(base_dir, 'binaries')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(bin_dir, exist_ok=True)
        
        # --- Generate Files ---
        # Log
        log_path = os.path.join(log_dir, f'{incident_id}.csv')
        generate_dummy_log(log_path)
        
        # Images
        for i in range(config['timed_imgs']):
            generate_dummy_image(os.path.join(img_dir, f'timed_{i}.jpg'), add_timestamp=True)
        for i in range(config['static_imgs']):
            generate_dummy_image(os.path.join(img_dir, f'static_{i}.jpg'), add_timestamp=False)
            
        # Binary
        bin_path = ""
        if config['has_binary']:
            bin_path = os.path.join(bin_dir, f'{incident_id}.exe')
            generate_dummy_binary(bin_path)

        # --- Add to Manifest ---
        manifest_data.append({
            'incident_id': incident_id,
            'attack_type': config['attack_type'],
            'log_path': os.path.relpath(log_path, base_dir),
            'image_folder': os.path.relpath(img_dir, base_dir),
            'binary_path': os.path.relpath(bin_path, base_dir) if bin_path else ""
        })

    # --- Create Manifest CSV ---
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = os.path.join(base_dir, 'manifest.csv')
    manifest_df.to_csv(manifest_path, index=False)
    
    print("\n--- Dataset Generation Complete ---")
    print(f"Manifest file created at: {manifest_path}")
    print("Subdirectories 'logs', 'images', and 'binaries' created in 'data/'.")
    print("\nTo test the full pipeline, you can now try to run the training script.")

if __name__ == '__main__':
    main()
