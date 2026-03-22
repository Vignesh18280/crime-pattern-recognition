"""
train.py
--------
Training pipeline for the Hybrid Siamese CNN Bi-LSTM model.

Handles:
    - Loss function (Binary Cross Entropy)
    - Optimizer (Adam)
    - Training loop with validation
    - Accuracy and loss tracking
    - Model checkpointing (saves best model)
    - Results visualization (accuracy + loss curves)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from dataset import build_dataloaders
from model   import SiameseCrimeMatcher, print_model_summary


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

RESULTS_DIR  = "results"
CHECKPOINT   = "results/best_model.pth"
EPOCHS       = 20
BATCH_SIZE   = 32
LEARNING_RATE= 0.001
FEATURE_DIM  = 64
SEQ_LEN      = 5


# ─── LOSS FUNCTION ────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.

    For SAME MO pairs   (label=1): penalizes large distance
    For DIFFERENT pairs (label=0): penalizes small distance

    We also use BCE loss combined for stable training.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin  = margin
        self.bce     = nn.BCELoss()

    def forward(self, similarity, label):
        """
        similarity : model output (0.0 - 1.0)
        label      : 1 = same MO, 0 = different MO
        """
        return self.bce(similarity, label)


# ─── ACCURACY ────────────────────────────────────────────────────────────────

def compute_accuracy(scores, labels, threshold=0.5):
    """
    Converts similarity scores to binary predictions.
    Score >= threshold → predicted same MO (1)
    Score <  threshold → predicted different MO (0)
    """
    predictions = (scores >= threshold).float()
    correct     = (predictions == labels).float().sum()
    return (correct / len(labels)).item()


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one full pass through the training data.
    Returns average loss and accuracy for this epoch.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for batch_a, batch_b, labels in tqdm(loader, desc="  Training", leave=False):
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)
        labels  = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        scores = model(batch_a, batch_b)

        # Compute loss
        loss = criterion(scores, labels)

        # Backward pass — this is where the model learns
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_acc  += compute_accuracy(scores.detach(), labels.detach())

    avg_loss = total_loss / len(loader)
    avg_acc  = total_acc  / len(loader)
    return avg_loss, avg_acc


# ─── VALIDATION LOOP ──────────────────────────────────────────────────────────

def validate(model, loader, criterion, device):
    """
    Evaluates model on validation set without updating weights.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for batch_a, batch_b, labels in tqdm(loader, desc="  Validating", leave=False):
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            labels  = labels.to(device)

            scores = model(batch_a, batch_b)
            loss   = criterion(scores, labels)

            total_loss += loss.item()
            total_acc  += compute_accuracy(scores, labels)

    avg_loss = total_loss / len(loader)
    avg_acc  = total_acc  / len(loader)
    return avg_loss, avg_acc


# ─── PLOT RESULTS ─────────────────────────────────────────────────────────────

def plot_training_curves(history):
    """
    Plots and saves:
    1. Training vs Validation Accuracy curve
    2. Training vs Validation Loss curve
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sns.set_style("whitegrid")
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hybrid Siamese CNN Bi-LSTM — Training Results",
                 fontsize=14, fontweight="bold")

    # ── Accuracy plot ──
    ax1.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy",
             linewidth=2, markersize=5)
    ax1.plot(epochs, history["val_acc"],   "r-o", label="Val Accuracy",
             linewidth=2, markersize=5)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # ── Loss plot ──
    ax2.plot(epochs, history["train_loss"], "b-o", label="Train Loss",
             linewidth=2, markersize=5)
    ax2.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",
             linewidth=2, markersize=5)
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[train] Training curves saved → {path}")


def plot_similarity_distribution(model, val_loader, device):
    """
    Plots the distribution of similarity scores for:
    - Same MO pairs    (should cluster near 1.0)
    - Different pairs  (should cluster near 0.0)

    This visually proves the model learned to separate MOs.
    """
    model.eval()
    same_scores, diff_scores = [], []

    with torch.no_grad():
        for batch_a, batch_b, labels in val_loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            scores  = model(batch_a, batch_b).cpu().numpy()
            labels  = labels.numpy()

            same_scores.extend(scores[labels == 1])
            diff_scores.extend(scores[labels == 0])

    plt.figure(figsize=(10, 5))
    plt.suptitle("Similarity Score Distribution", fontsize=13, fontweight="bold")

    sns.histplot(same_scores, color="green", label="Same MO (should → 1.0)",
                 alpha=0.6, bins=30, kde=True)
    sns.histplot(diff_scores, color="red",   label="Different MO (should → 0.0)",
                 alpha=0.6, bins=30, kde=True)

    plt.axvline(x=0.5, color="black", linestyle="--", label="Decision boundary (0.5)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.legend()

    path = os.path.join(RESULTS_DIR, "similarity_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Similarity distribution saved → {path}")


# ─── MAIN TRAINING PIPELINE ───────────────────────────────────────────────────

def train():
    """
    Full training pipeline:
    1. Load data
    2. Build model
    3. Train for EPOCHS
    4. Save best model
    5. Plot results
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Using device: {device}")

    # ── 1. Data ──
    print("\n[train] Loading data...")
    train_loader, val_loader, feat_dim, n_cls, cls_names = build_dataloaders(
        batch_size = BATCH_SIZE,
        seq_len    = SEQ_LEN
    )

    # ── 2. Model ──
    print("\n[train] Building model...")
    model = SiameseCrimeMatcher(
        feature_dim = feat_dim,
        cnn_out_dim = 128,
        mo_dim      = 256
    ).to(device)
    print_model_summary(model, feature_dim=feat_dim, seq_len=SEQ_LEN)

    # ── 3. Loss + Optimizer ──
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # ── 4. Training Loop ──
    print(f"\n[train] Starting training for {EPOCHS} epochs...\n")

    history    = {"train_loss": [], "train_acc": [],
                  "val_loss":   [], "val_acc":   []}
    best_val   = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(
            model, val_loader, criterion, device)

        scheduler.step()

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print epoch summary
        print(f"  Train → Loss: {train_loss:.4f}  Acc: {train_acc:.2%}")
        print(f"  Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.2%}")

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "val_loss":    val_loss,
                "cls_names":   cls_names,
                "feat_dim":    feat_dim,
            }, CHECKPOINT)
            print(f"  ✅ Best model saved (val_acc: {val_acc:.2%})")

        print()

    elapsed = time.time() - start_time
    print(f"[train] Training complete in {elapsed/60:.1f} minutes")
    print(f"[train] Best validation accuracy: {best_val:.2%}")

    # ── 5. Plot Results ──
    plot_training_curves(history)
    plot_similarity_distribution(model, val_loader, device)

    return model, history, val_loader, device


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, history, val_loader, device = train()
    print("\n[train] All done! Check results/ folder for plots.")