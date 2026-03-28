"""
train.py
--------
Training pipeline for the Hybrid Siamese CNN Bi-LSTM model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from dataset import build_dataloaders
from model   import SiameseCrimeMatcher, print_model_summary

# ─── CONSTANTS (FIXED) ────────────────────────────────────────────────────────

RESULTS_DIR   = "results"
CHECKPOINT    = "results/best_model.pth"
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 0.0003   # fixed: was too high at 0.001
FEATURE_DIM   = 32       # matches dataset.py
SEQ_LEN       = 5


# ─── ACCURACY ─────────────────────────────────────────────────────────────────

def compute_accuracy(scores, labels, threshold=0.5):
    predictions = (scores >= threshold).float()
    correct     = (predictions == labels).float().sum()
    return (correct / len(labels)).item()


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for batch_a, batch_b, labels in tqdm(loader, desc="  Training", leave=False):
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        scores = model(batch_a, batch_b)
        loss   = criterion(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(scores.detach(), labels.detach())

    return total_loss / len(loader), total_acc / len(loader)


# ─── VALIDATION LOOP ──────────────────────────────────────────────────────────

def validate(model, loader, criterion, device):
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

    return total_loss / len(loader), total_acc / len(loader)


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_training_curves(history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sns.set_style("whitegrid")
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hybrid Siamese CNN Bi-LSTM — Training Results",
                 fontsize=14, fontweight="bold")

    ax1.plot(epochs, history["train_acc"], "b-o", label="Train", linewidth=2, markersize=4)
    ax1.plot(epochs, history["val_acc"],   "r-o", label="Val",   linewidth=2, markersize=4)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax2.plot(epochs, history["train_loss"], "b-o", label="Train", linewidth=2, markersize=4)
    ax2.plot(epochs, history["val_loss"],   "r-o", label="Val",   linewidth=2, markersize=4)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[train] Training curves saved → {path}")


def plot_similarity_distribution(model, val_loader, device):
    model.eval()
    same_scores, diff_scores = [], []

    with torch.no_grad():
        for batch_a, batch_b, labels in val_loader:
            scores = model(batch_a.to(device), batch_b.to(device)).cpu().numpy()
            labels = labels.numpy()
            same_scores.extend(scores[labels == 1])
            diff_scores.extend(scores[labels == 0])

    plt.figure(figsize=(10, 5))
    plt.suptitle("Similarity Score Distribution — Same MO vs Different MO",
                 fontsize=13, fontweight="bold")
    sns.histplot(same_scores, color="green", label="Same MO (→ 1.0)",
                 alpha=0.6, bins=30, kde=True)
    sns.histplot(diff_scores, color="red",   label="Different MO (→ 0.0)",
                 alpha=0.6, bins=30, kde=True)
    plt.axvline(x=0.5, color="black", linestyle="--", label="Decision boundary")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.legend()

    path = os.path.join(RESULTS_DIR, "similarity_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Similarity distribution saved → {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device: {device}")

    # Data
    train_loader, val_loader, feat_dim, n_cls, cls_names = build_dataloaders(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # Model — smaller for our dataset size
    model = SiameseCrimeMatcher(
        feature_dim = feat_dim,
        cnn_out_dim = 64,     # reduced from 128
        mo_dim      = 128     # reduced from 256
    ).to(device)
    print_model_summary(model, feature_dim=feat_dim, seq_len=SEQ_LEN)

    # Loss + optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    print(f"\n[train] Training for {EPOCHS} epochs...\n")
    history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    start    = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train → Loss: {train_loss:.4f}  Acc: {train_acc:.2%}")
        print(f"  Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.2%}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "feat_dim":    feat_dim,
                "cls_names":   cls_names,
                "cnn_out_dim": 64,
                "mo_dim":      128,
            }, CHECKPOINT)
            print(f"  ✅ Best model saved (val_acc: {val_acc:.2%})")
        print()

    elapsed = time.time() - start
    print(f"[train] Done in {elapsed/60:.1f} min | Best val acc: {best_val:.2%}")

    plot_training_curves(history)
    plot_similarity_distribution(model, val_loader, device)

    return model, history, val_loader, device


if __name__ == "__main__":
    model, history, val_loader, device = train()
    print("\n[train] Check results/ folder for plots!")
