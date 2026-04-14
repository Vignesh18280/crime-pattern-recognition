"""
Script to update manifest.csv with real Malimg images.
Run after downloading Malimg dataset.

Usage:
    python3 update_manifest_with_malimg.py
"""

import os
import csv

DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

def update_manifest():
    """Update manifest with Malimg image paths."""
    
    # Malimg attack type mapping
    malimg_mapping = {
        "Allaple.A": "worm",
        "Allaple.L": "worm", 
        "Agent.FYI": "backdoor",
        "Obfuscator.AD": "trojan",
        "Fakerean": "rogue"
    }
    
    new_entries = []
    
    # Check for existing malimg images
    images_dir = os.path.join(DATA_DIR, "images")
    if os.path.exists(images_dir):
        for folder in os.listdir(images_dir):
            folder_path = os.path.join(images_dir, folder)
            if os.path.isdir(folder_path) and folder.startswith("malimg_"):
                # Extract family name
                family = folder.replace("malimg_", "")
                attack_type = malimg_mapping.get(family, "malware")
                
                new_entries.append({
                    "incident_id": folder,
                    "attack_type": attack_type,
                    "log_path": "",
                    "image_folder": f"images/{folder}",
                    "binary_path": ""
                })
                print(f"Found: {folder} ({attack_type})")
    
    if not new_entries:
        print("[update] No Malimg images found. Run download_malimg.py first on Colab.")
        return
    
    # Read existing manifest
    existing = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    
    # Combine
    all_entries = existing + new_entries
    
    # Write updated manifest
    with open(MANIFEST_PATH, 'w', newline='') as f:
        fieldnames = ["incident_id", "attack_type", "log_path", "image_folder", "binary_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)
    
    print(f"\n[update] Updated manifest with {len(new_entries)} new Malimg entries")
    print(f"[update] Total entries: {len(all_entries)}")


if __name__ == "__main__":
    update_manifest()