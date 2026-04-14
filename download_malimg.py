"""
Download datasets for multi-modal crime pattern recognition.
Run on Google Colab: python3 download_malimg.py

Uses Kaggle API - set KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets.
"""

import os
import csv

DATA_DIR = "data"


def download_with_kaggle():
    """Download datasets using Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"[download] Kaggle auth failed: {e}")
        print("[download] Set KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets")
        return None


def download_malimg(api):
    """Download Malimg malware dataset."""
    malimg_dir = os.path.join(DATA_DIR, "malimg")
    
    if os.path.exists(malimg_dir):
        print(f"[download] Malimg exists: {malimg_dir}")
        return
    
    print("[download] Downloading Malimg dataset...")
    try:
        api.dataset_download_files('ikrambenabd/malimg-original', path=DATA_DIR, unzip=True)
        
        # Find and rename extracted folder
        for f in os.listdir(DATA_DIR):
            if 'malimg' in f.lower() and os.path.isdir(os.path.join(DATA_DIR, f)):
                os.rename(os.path.join(DATA_DIR, f), malimg_dir)
                break
        
        print(f"[download] Downloaded: {malimg_dir}")
    except Exception as e:
        print(f"[download] Malimg failed: {e}")


def download_weapons(api):
    """Download weapon detection dataset."""
    weapons_dir = os.path.join(DATA_DIR, "images", "weapons")
    
    if os.path.exists(weapons_dir):
        print(f"[download] Weapons exists")
        return
    
    print("[download] Downloading weapon dataset...")
    try:
        # Roboflow format needs different approach, try Kaggle alternatives
        api.dataset_download_files('zinkzsa/weapon-and-knife-detection-137k-images', 
                                   path=DATA_DIR, unzip=True)
        print(f"[download] Downloaded weapons")
    except Exception as e:
        print(f"[download] Weapons failed: {e}")


def create_crime_scene_images():
    """Create placeholder crime scene images."""
    from PIL import Image
    
    crime_types = {
        "knife_scene": "knife",
        "gun_scene": "firearm", 
        "weapon_scene": "weapon"
    }
    
    for scene_name, attack_type in crime_types.items():
        scene_dir = os.path.join(DATA_DIR, "images", scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        for i in range(3):
            img = Image.new('RGB', (256, 256), color=(100 + i*20, 50, 50))
            img.save(os.path.join(scene_dir, f"scene_{i}.jpg"))
        
        print(f"[download] Created: {scene_name}")
    
    return list(crime_types.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Modal Dataset Setup")
    print("=" * 60)
    
    # Try Kaggle download
    print("\n--- Kaggle Download ---")
    api = download_with_kaggle()
    
    if api:
        download_malimg(api)
        # download_weapons(api)  # Optional - larger dataset
    else:
        print("[download] Using placeholder images")
        create_crime_scene_images()
    
    # Update manifest
    print("\n--- Updating manifest ---")
    manifest_path = os.path.join(DATA_DIR, "manifest.csv")
    
    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            existing = list(csv.DictReader(f))
    
    # Check what images exist
    images_dir = os.path.join(DATA_DIR, "images")
    crime_entries = []
    if os.path.exists(images_dir):
        for folder in os.listdir(images_dir):
            folder_path = os.path.join(images_dir, folder)
            if os.path.isdir(folder_path):
                imgs = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
                if imgs:
                    crime_entries.append({
                        "incident_id": folder,
                        "attack_type": folder.split('_')[0] if '_' in folder else "crime",
                        "log_path": "",
                        "image_folder": f"images/{folder}",
                        "binary_path": ""
                    })
    
    all_entries = existing + crime_entries
    
    with open(manifest_path, 'w', newline='') as f:
        fieldnames = ["incident_id", "attack_type", "log_path", "image_folder", "binary_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)
    
    print(f"\n[download] Total: {len(all_entries)} entries")
    print("[download] Run: python3 src/train.py")