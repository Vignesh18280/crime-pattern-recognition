"""
explain.py
----------
Explainable AI (XAI) for the Hybrid Siamese CNN Bi-LSTM model.

Uses SHAP to answer the critical forensic question:
"WHY did the model say these two crimes have the same MO?"

This is essential for:
    - Legal admissibility of AI evidence
    - Investigator trust and validation
    - Academic credibility of the system

Output:
    - Feature importance bar chart
    - SHAP summary plot
    - Per-pair explanation report
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Add src/ to path so imports work from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import build_dataloaders, FEATURE_DIM
from model   import SiameseCrimeMatcher

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
CHECKPOINT  = "results/best_model.pth"
SEQ_LEN     = 5

# Real UNSW-NB15 feature names (first 32 features used)
FEATURE_NAMES = [
    "duration",      "src_packets",   "dst_packets",   "src_bytes",
    "dst_bytes",     "src_load",      "dst_load",      "src_loss",
    "dst_loss",      "src_jitter",    "dst_jitter",    "src_win",
    "dst_win",       "src_ttl",       "dst_ttl",       "tcp_rtt",
    "syn_ack",       "ack_dat",       "src_mean_pkt",  "dst_mean_pkt",
    "trans_depth",   "res_bdy_len",   "src_inter_pkt", "dst_inter_pkt",
    "ct_state_ttl",  "ct_flw_http",   "ct_ftp_cmd",    "ct_srv_src",
    "ct_srv_dst",    "ct_dst_ltm",    "ct_src_ltm",    "ct_src_dport"
]


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path=CHECKPOINT):
    """
    Loads the best trained Siamese model from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run train.py first."
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feat_dim    = ckpt.get("feat_dim",    FEATURE_DIM)
    cnn_out_dim = ckpt.get("cnn_out_dim", 64)
    mo_dim      = ckpt.get("mo_dim",      128)
    cls_names   = ckpt.get("cls_names",   [])

    model = SiameseCrimeMatcher(
        feature_dim = feat_dim,
        cnn_out_dim = cnn_out_dim,
        mo_dim      = mo_dim
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[explain] Model loaded from {checkpoint_path}")
    print(f"[explain] Val accuracy at save: {ckpt['val_acc']:.2%}")
    print(f"[explain] Classes: {cls_names}")
    return model, feat_dim, cls_names


# ─── WRAPPER FOR SHAP ─────────────────────────────────────────────────────────

class SiameseWrapper:
    """
    Wraps the Siamese model for SHAP compatibility.

    SHAP needs a function that takes a flat numpy array and returns
    a scalar prediction. We fix pattern_B and vary pattern_A to
    understand which features of pattern_A drive the similarity score.
    """

    def __init__(self, model, fixed_pattern_b, feat_dim, seq_len):
        self.model     = model
        self.pattern_b = torch.tensor(
            fixed_pattern_b, dtype=torch.float32).unsqueeze(0)
        self.feat_dim  = feat_dim
        self.seq_len   = seq_len

    def predict(self, X):
        """
        X: numpy array of shape (n_samples, seq_len * feat_dim)
        Returns: similarity scores of shape (n_samples,)
        """
        results = []
        for row in X:
            pattern_a = torch.tensor(
                row.reshape(self.seq_len, self.feat_dim),
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                score = self.model(pattern_a, self.pattern_b)
            results.append(score.item())

        return np.array(results)


# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

def compute_feature_importance(model, val_loader, feat_dim, n_samples=50):
    """
    Uses SHAP KernelExplainer to compute feature importance.

    Fixes one crime pattern (pattern_B) and measures how much
    each feature of pattern_A contributes to the similarity score.

    Returns:
        shap_values  : array of SHAP values per feature
        feature_names: list of feature name strings
    """
    print("[explain] Computing SHAP feature importance...")
    print("[explain] This may take 1-2 minutes...")

    # Get a batch of validation data
    batch_a, batch_b, labels = next(iter(val_loader))

    # Use first sample as fixed reference (pattern_B)
    fixed_b = batch_b[0].numpy()   # (seq_len, feat_dim)

    # Create wrapper
    wrapper = SiameseWrapper(model, fixed_b, feat_dim, SEQ_LEN)

    # Flatten pattern_A samples for SHAP: (n, seq_len * feat_dim)
    n_samples = min(n_samples, len(batch_a))
    X_flat = batch_a[:n_samples].numpy().reshape(n_samples, -1)

    # Background dataset for SHAP (small subset)
    background = X_flat[:min(10, len(X_flat))]

    # KernelExplainer — model-agnostic, works with any black box
    explainer   = shap.KernelExplainer(wrapper.predict, background)
    shap_values = explainer.shap_values(X_flat[:min(20, len(X_flat))], nsamples=50)

    print(f"[explain] SHAP values computed. Shape: {shap_values.shape}")
    return shap_values, X_flat[:20]


# ─── AGGREGATE FEATURE NAMES ──────────────────────────────────────────────────

def get_flat_feature_names(feat_dim, seq_len):
    """
    Creates feature names for the flattened sequence input.
    Format: "t1_duration", "t1_src_bytes", ..., "t5_ct_src_dport"
    """
    names = []
    for t in range(1, seq_len + 1):
        for f in FEATURE_NAMES[:feat_dim]:
            names.append(f"t{t}_{f}")
    return names


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_feature_importance(shap_values, feat_dim, top_n=20):
    """
    Plots top N most important features by mean absolute SHAP value.

    This answers: "Which network features does the model rely on
    most to determine if two crimes share the same MO?"
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    flat_names  = get_flat_feature_names(feat_dim, SEQ_LEN)
    mean_shap   = np.abs(shap_values).mean(axis=0)

    # Get top N features
    top_idx     = np.argsort(mean_shap)[::-1][:top_n]
    top_vals    = mean_shap[top_idx]
    top_names   = [flat_names[i] if i < len(flat_names)
                   else f"feature_{i}" for i in top_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = sns.color_palette("RdYlGn_r", top_n)

    bars = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| — Feature Importance")
    ax.set_title(
        f"Top {top_n} Features Driving MO Similarity\n"
        "Hybrid Siamese CNN Bi-LSTM — XAI Explanation",
        fontsize=12, fontweight="bold"
    )
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Feature importance plot saved → {path}")


def plot_shap_summary(shap_values, X_flat, feat_dim):
    """
    SHAP summary plot — shows direction of feature impact.
    Red = high feature value increases similarity
    Blue = high feature value decreases similarity
    """
    flat_names = get_flat_feature_names(feat_dim, SEQ_LEN)

    # Trim to available names
    n_features = shap_values.shape[1]
    names      = (flat_names + [f"feature_{i}" for i in range(len(flat_names), n_features)])[:n_features]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_flat,
        feature_names = names,
        max_display   = 15,
        show          = False,
        plot_type     = "dot"
    )
    plt.title("SHAP Summary — Feature Impact on MO Similarity Score",
              fontsize=12, fontweight="bold")

    path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] SHAP summary plot saved → {path}")


# ─── LIVE DEMO — EXPLAIN ONE PAIR ─────────────────────────────────────────────

def explain_pair(model, pattern_a, pattern_b, feat_dim, class_names):
    """
    Explains the similarity between ONE specific pair of crime patterns.

    Outputs:
        - Similarity score
        - Verdict (same MO / different MO)
        - Top 5 features that drove the decision
    """
    print("\n" + "=" * 55)
    print("  CRIME PATTERN COMPARISON — XAI EXPLANATION")
    print("=" * 55)

    # Get similarity score
    with torch.no_grad():
        ta = torch.tensor(pattern_a, dtype=torch.float32).unsqueeze(0)
        tb = torch.tensor(pattern_b, dtype=torch.float32).unsqueeze(0)
        score = model(ta, tb).item()

    verdict = "SAME MO — Likely same threat actor" if score >= 0.5 \
              else "DIFFERENT MO — Unrelated crimes"

    print(f"\n  Similarity Score : {score:.4f}")
    print(f"  Verdict          : {verdict}")
    print(f"  Confidence       : {max(score, 1-score):.2%}")

    # Compute quick feature differences
    mean_a = pattern_a.mean(axis=0)   # average feature values across sequence
    mean_b = pattern_b.mean(axis=0)

    diff   = np.abs(mean_a - mean_b)
    top5   = np.argsort(diff)[::-1][:5]

    print(f"\n  Top 5 most different features between crimes:")
    print(f"  {'Feature':<25} {'Crime A':>10} {'Crime B':>10} {'Diff':>10}")
    print(f"  {'-'*55}")

    for idx in top5:
        fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        print(f"  {fname:<25} {mean_a[idx]:>10.4f} {mean_b[idx]:>10.4f} {diff[idx]:>10.4f}")

    print("=" * 55)
    return score


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_explanations():
    """
    Full XAI pipeline:
    1. Load trained model
    2. Load validation data
    3. Compute SHAP values
    4. Plot feature importance
    5. Demo pair explanation
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load model
    model, feat_dim, cls_names = load_model()

    # 2. Load data
    print("\n[explain] Loading validation data...")
    _, val_loader, _, _, _ = build_dataloaders(batch_size=32, seq_len=SEQ_LEN)

    # 3. SHAP computation
    shap_values, X_flat = compute_feature_importance(model, val_loader, feat_dim)

    # 4. Plots
    plot_feature_importance(shap_values, feat_dim)
    plot_shap_summary(shap_values, X_flat, feat_dim)

    # 5. Live pair explanation
    batch_a, batch_b, labels = next(iter(val_loader))

    print("\n[explain] === SAME MO PAIR (label=1) ===")
    same_idx = (labels == 1).nonzero(as_tuple=True)[0][0].item()
    explain_pair(
        model,
        batch_a[same_idx].numpy(),
        batch_b[same_idx].numpy(),
        feat_dim,
        cls_names
    )

    print("\n[explain] === DIFFERENT MO PAIR (label=0) ===")
    diff_idx = (labels == 0).nonzero(as_tuple=True)[0][0].item()
    explain_pair(
        model,
        batch_a[diff_idx].numpy(),
        batch_b[diff_idx].numpy(),
        feat_dim,
        cls_names
    )

    print("\n[explain] All done! Check results/ folder:")
    print("  → results/feature_importance.png")
    print("  → results/shap_summary.png")
    print("  → results/training_curves.png")
    print("  → results/similarity_distribution.png")


if __name__ == "__main__":
    run_explanations()