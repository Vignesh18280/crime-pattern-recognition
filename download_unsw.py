"""
Download UNSW-NB15 dataset from HuggingFace.
Run on Google Colab: python3 download_unsw.py
"""

import os

DATA_DIR = "data"

def download_unsw_nb15():
    """Download UNSW-NB15 dataset from HuggingFace."""
    train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.parquet")
    test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.parquet")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"[download] UNSW-NB15 already exists")
        return
    
    print("[download] Loading UNSW-NB15 from HuggingFace...")
    from datasets import load_dataset
    
    # Load temporal (standard) split - ~175K train, ~82K test
    ds = load_dataset("lacg030175/UNSW-NB15", "temporal")
    
    # Save as parquet
    ds["train"].to_parquet(train_path)
    ds["test"].to_parquet(test_path)
    
    print(f"[download] Saved: {len(ds['train'])} train, {len(ds['test'])} test rows")


if __name__ == "__main__":
    print("=" * 60)
    print("Download UNSW-NB15 from HuggingFace")
    print("=" * 60)
    
    # Install dependencies if needed
    try:
        import datasets
    except ImportError:
        print("Installing datasets...")
        import subprocess
        subprocess.run(["pip", "install", "datasets", "pandas", "pyarrow"], check=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    download_unsw_nb15()
    print("\n[download] Done! Run: python3 download_malimg.py")