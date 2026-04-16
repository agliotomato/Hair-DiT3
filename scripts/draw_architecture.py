"""
Architecture diagram for Hair-DiT3 (EHC v2).
Generates architecture.png in the project root.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ─── canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# ─── color palette ────────────────────────────────────────────────────────────
C = {
    "input":      "#16213e",
    "frozen":     "#0f3460",
    "trainable":  "#533483",
    "module":     "#e94560",
    "loss":       "#2d6a4f",
    "aug":        "#1b4332",
    "comp":       "#b5451b",
    "arrow":      "#e0e0e0",
    "text":       "#f0f0f0",
    "dim":        "#a0c4ff",
    "border_frozen":    "#4fc3f7",
    "border_train":     "#ce93d8",
    "border_module":    "#ef9a9a",
    "border_loss":      "#80cbc4",
    "border_comp":      "#ffcc80",
}

def box(ax, x, y, w, h, label, sublabel="", color=C["input"], border=C["border_frozen"],
        fontsize=10, subfontsize=8, radius=0.3):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor=border, linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    cy = y + h/2 + (0.15 if sublabel else 0)
    ax.text(x + w/2, cy, label,
            ha="center", va="center", color=C["text"],
            fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", color=C["dim"],
                fontsize=subfontsize, zorder=4, style="italic")

def arrow(ax, x1, y1, x2, y2, label="", color=C["arrow"], lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, color=C["dim"],
                fontsize=7.5, zorder=6, style="italic")

def section_label(ax, x, y, text, color="#ffffff"):
    ax.text(x, y, text, color=color, fontsize=9, fontweight="bold",
            ha="left", va="center", zorder=6,
            bbox=dict(facecolor="#00000055", edgecolor="none", pad=2))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(11, 13.5, "Hair-DiT3  EHC v2  Architecture",
        ha="center", va="center", color="#ffffff",
        fontsize=16, fontweight="bold", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Inputs  (y=11.2)
# ══════════════════════════════════════════════════════════════════════════════
# Background
box(ax, 0.4,  11.2, 2.2, 0.9, "Background", "[B,3,512,512]",
    C["input"], C["border_frozen"])
# Sketch (RGB)
box(ax, 3.2,  11.2, 2.2, 0.9, "Sketch (RGB)", "[B,3,512,512]",
    C["input"], C["border_frozen"])
# Matte
box(ax, 6.0,  11.2, 2.2, 0.9, "Matte (α)", "[B,1,512,512]",
    C["input"], C["border_frozen"])
# Target (train only)
box(ax, 8.8,  11.2, 2.2, 0.9, "Target", "[B,3,512,512]  train",
    C["input"], "#888888")

section_label(ax, 0.4, 13.0, "① Inputs")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Encoders  (y=9.2)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.4, 10.8, "② Encoding  (ctrl_cond 구성)")

# VAE (background)
box(ax, 0.4,  9.2, 2.2, 0.85, "VAE encode", "z_bg  [B,16,64,64]",
    C["frozen"], C["border_frozen"])
# VAE (sketch)
box(ax, 3.2,  9.2, 2.2, 0.85, "VAE encode ❄", "sketch_lat [B,16,64,64]",
    C["frozen"], C["border_frozen"])
# MatteCNN
box(ax, 6.0,  9.2, 2.2, 0.85, "MatteCNN ✦", "matte_feat [B,16,64,64]",
    C["trainable"], C["border_train"])
# bilinear
box(ax, 8.8,  9.2, 2.2, 0.85, "bilinear ↓", "matte_lat  [B,1,64,64]",
    C["input"], "#888888")
# VAE (target)
box(ax, 11.6, 9.2, 2.2, 0.85, "VAE encode", "z_target [B,16,64,64]",
    C["frozen"], C["border_frozen"])

# arrows row1→row2
arrow(ax, 1.5, 11.2, 1.5, 10.05)
arrow(ax, 4.3, 11.2, 4.3, 10.05)
arrow(ax, 7.1, 11.2, 7.1, 10.05)
arrow(ax, 7.1, 11.2, 9.9, 10.05)
arrow(ax, 9.9, 11.2, 12.7, 10.05)

# ══════════════════════════════════════════════════════════════════════════════
# ctrl_cond fusion box  (y=7.7)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.4, 8.85, "③ ctrl_cond  fusion")

box(ax, 3.2,  7.7, 5.8, 0.9, "sketch_latent  +  matte_feat  ‖  matte_latent",
    "ctrl_cond  [B, 17, 64, 64]",
    C["module"], C["border_module"], fontsize=9.5)

arrow(ax, 4.3, 9.2, 5.0, 8.6, "+")
arrow(ax, 7.1, 9.2, 6.5, 8.6)
arrow(ax, 9.9, 9.2, 8.5, 8.6, "cat")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — ControlNet + Blending  (y=6.0)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.4, 7.4, "④ SD3 ControlNet  (12 layers, Zero-init)")

box(ax, 0.4,  6.0, 3.0, 1.0,
    "SD3ControlNet ✦", "residuals  [B,1024,1152] × 12",
    C["trainable"], C["border_train"])
arrow(ax, 6.1, 7.7, 2.3, 7.0, "ctrl_cond")

# Matte Tokenizer
box(ax, 4.0,  6.0, 2.4, 1.0,
    "MattePatchTokenizer", "matte_tokens [B,1024,1]",
    C["input"], "#888888", fontsize=8.5)
arrow(ax, 9.9, 9.2, 5.2, 7.0, "matte_lat")

# Blending
box(ax, 7.0,  6.0, 3.2, 1.0,
    "Matte-Gated Blend", "early: r   /   late: r × tokens",
    C["module"], C["border_module"], fontsize=9)
arrow(ax, 3.4,  6.5, 7.0,  6.5, "residuals")
arrow(ax, 6.4,  6.5, 7.0,  6.5, "tokens")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — MM-DiT + CFG  (y=4.1)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.4, 5.7, "⑤ MM-DiT Transformer  (Frozen)  +  CFG")

# z_noisy
box(ax, 0.4,  4.1, 2.0, 0.9, "z_noisy", "(1-σ)·z + σ·ε",
    C["input"], "#888888", fontsize=9)

# Null Embeddings
box(ax, 2.8,  4.1, 2.4, 0.9, "Null Embeddings ✦", "[333,4096] / [2048]",
    C["trainable"], C["border_train"], fontsize=8.5)

# Transformer cond
box(ax, 5.6,  4.1, 3.0, 0.9, "Transformer ❄  (cond)", "noise_pred_cond",
    C["frozen"], C["border_frozen"], fontsize=9)
arrow(ax, 8.6,  6.5, 7.2,  5.0, "blended")

# Transformer uncond
box(ax, 9.0,  4.1, 3.0, 0.9, "Transformer ❄  (uncond)", "noise_pred_uncond",
    C["frozen"], C["border_frozen"], fontsize=9)

arrow(ax, 1.4,  4.1, 1.4,  3.3)  # z_noisy down
arrow(ax, 1.4,  4.1, 7.1,  5.0)  # z_noisy → cond
arrow(ax, 1.4,  4.1, 10.5, 5.0)  # z_noisy → uncond

# CFG box
box(ax, 5.6,  2.9, 6.4, 0.9,
    "CFG  +  Rescaled CFG",
    "noise_pred = uncond + s·(cond−uncond)   rescale by std ratio",
    C["module"], C["border_module"], fontsize=9)
arrow(ax, 7.1, 4.1, 7.1, 3.8)
arrow(ax, 10.5, 4.1, 10.5, 3.8)
arrow(ax, 3.8, 4.1, 3.8, 3.5, "null emb")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Scheduler + Compositor  (y=1.3)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.4, 2.6, "⑥ Scheduler step  →  ⑦ Compositor")

box(ax, 4.0,  1.3, 3.2, 1.0, "scheduler.step", "z  [B,16,64,64]",
    C["input"], "#888888")
arrow(ax, 8.8, 2.9, 5.6, 2.3, "noise_pred")

box(ax, 7.8,  1.3, 3.6, 1.0,
    "TimestepAwareCompositor",
    "z·m̃ + z_bg^(σ)·(1−m̃)   blur∝(1−σ)",
    C["comp"], C["border_comp"], fontsize=8.5)
arrow(ax, 7.2, 1.8, 7.8, 1.8)
arrow(ax, 1.5, 9.2, 1.5, 2.0)  # z_bg long arrow

# Loop back (dashed)
ax.annotate("", xy=(3.4, 4.5), xytext=(11.4, 1.8),
            arrowprops=dict(arrowstyle="-|>", color="#ffcc80",
                            lw=1.4, linestyle="dashed", mutation_scale=12),
            zorder=5)
ax.text(13.0, 3.1, "× 28 steps\n(denoising loop)",
        color="#ffcc80", fontsize=8.5, ha="left", va="center", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# VAE Decode → output
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 11.8, 1.3, 2.4, 1.0, "VAE decode", "Output  [B,3,H,W]",
    C["frozen"], C["border_frozen"])
arrow(ax, 11.4, 1.8, 11.8, 1.8)

# ══════════════════════════════════════════════════════════════════════════════
# LOSS (right side, y=6~9)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 15.0, 9.5, "⑧ Loss  (train only)")

box(ax, 15.0, 8.5, 6.5, 0.75, "L_flow  =  masked MSE(noise_pred, noise−z_target)",
    "헤어=1.0, 배경=0.1 weight   (λ=1.0)",
    C["loss"], C["border_loss"], fontsize=8.5, subfontsize=7.5)

box(ax, 15.0, 7.5, 6.5, 0.75, "L_bg  =  MSE(pred_z0·(1−m),  z_bg·(1−m))",
    "배경 latent 보존   (λ=3.0)",
    C["loss"], C["border_loss"], fontsize=8.5, subfontsize=7.5)

box(ax, 15.0, 6.5, 6.5, 0.75, "L_lpips  =  LPIPS-VGG  on hair region",
    "Phase1: step≥30%  /  Phase2: always   (λ=0.1)",
    C["loss"], C["border_loss"], fontsize=8.5, subfontsize=7.5)

box(ax, 15.0, 5.5, 6.5, 0.75, "L_edge  =  Sobel edge vs sketch stroke",
    "Phase2 only   (λ=0.05)",
    C["loss"], C["border_loss"], fontsize=8.5, subfontsize=7.5)

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION (right side, y=2.5~5.0)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 15.0, 5.1, "⑨ Augmentation  (train only)")

aug_items = [
    ("HFlip",                 "p=0.5   background / sketch / matte / target 동시 반전"),
    ("StrokeColorSampler",    "p=1.0   33% random pixel  /  67% mean pixel from hair"),
    ("ThicknessJitter",       "p=0.5   morphological dilation  (kernel 3×3)"),
    ("MatteBoundaryPerturb.", "p=0.3   elastic warp (amplitude=4, sigma=10)"),
]
for i, (name, desc) in enumerate(aug_items):
    y = 4.3 - i * 0.65
    box(ax, 15.0, y, 6.5, 0.55, name, desc,
        C["aug"], C["border_loss"], fontsize=8.5, subfontsize=7.5)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C["frozen"],    C["border_frozen"],  "❄ Frozen (VAE, Transformer)"),
    (C["trainable"], C["border_train"],   "✦ Trainable (ControlNet, MatteCNN, Null Emb)"),
    (C["comp"],      C["border_comp"],    "Compositor (no params)"),
    (C["loss"],      C["border_loss"],    "Loss"),
    (C["aug"],       C["border_loss"],    "Augmentation"),
]
for i, (fc, ec, lbl) in enumerate(legend_items):
    lx = 0.4 + i * 4.3
    rect = FancyBboxPatch((lx, 0.25), 0.45, 0.35,
                          boxstyle="round,pad=0.02",
                          facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.55, 0.43, lbl,
            color=C["text"], fontsize=8, va="center", zorder=4)

plt.tight_layout(pad=0.3)
out_path = "architecture.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
