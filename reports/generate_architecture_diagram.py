"""
Render an academic-style diagram of the dual encoder architecture.
Output: reports/dual_encoder_architecture.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- palette: white + pale faint blue only ---------------------------------
WHITE       = "#FFFFFF"
PALE_BLUE   = "#EAF2FA"   # primary fill
PALE_BLUE_2 = "#D4E3F2"   # accent fill
BORDER      = "#6C89AE"   # muted blue border
BORDER_SOFT = "#9FB5D1"   # subtle border
TEXT        = "#1C2536"   # near-black text

# ---- figure ----------------------------------------------------------------
FIG_W, FIG_H = 14.0, 19.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=240)
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


def box(cx, cy, w, h, text,
        fill=PALE_BLUE, border=BORDER, fontsize=10, weight="normal",
        rad=0.06):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rad}",
        facecolor=fill, edgecolor=border, linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=TEXT, weight=weight, family="serif")


def stack_box(cx, cy, w, h, text, depth=2, **kw):
    """Stack of identical boxes to suggest N repeated layers."""
    fill = kw.pop("fill", PALE_BLUE)
    for i in range(depth, 0, -1):
        patch = FancyBboxPatch(
            (cx - w / 2 + 0.08 * i, cy - h / 2 - 0.08 * i), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=fill, edgecolor=BORDER_SOFT, linewidth=1.0,
        )
        ax.add_patch(patch)
    box(cx, cy, w, h, text, fill=fill, **kw)


def arrow(x1, y1, x2, y2, color=BORDER, lw=1.1, style="-|>"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        shrinkA=3, shrinkB=3),
    )


# ---- title -----------------------------------------------------------------
ax.text(FIG_W / 2, 18.4,
        "Dual Encoder Architecture for Spectrum–Peptide Matching",
        ha="center", va="center", fontsize=15, weight="bold",
        family="serif", color=TEXT)
ax.text(FIG_W / 2, 17.9,
        "CLIP-style contrastive training over mass spectra and peptide sequences",
        ha="center", va="center", fontsize=10.5, style="italic",
        family="serif", color=TEXT)

# ---- branch headers --------------------------------------------------------
ax.text(3.0, 17.1, "Spectrum Encoder",
        ha="center", va="center", fontsize=12.5, weight="bold",
        family="serif", color=TEXT)
ax.text(11.0, 17.1, "Peptide Encoder",
        ha="center", va="center", fontsize=12.5, weight="bold",
        family="serif", color=TEXT)

# light vertical divider between columns
ax.plot([7.0, 7.0], [6.4, 16.8], color=BORDER_SOFT, lw=0.6, linestyle=(0, (2, 3)))

# ---- inputs ----------------------------------------------------------------
box(3.0, 16.3, 4.2, 0.7,
    "Mass spectrum   $[B,\\,500,\\,2]$   (m/z, intensity)",
    fill=PALE_BLUE_2, fontsize=10)

box(6.3, 16.3, 2.3, 0.7,
    "Precursor   $[B,\\,2]$\n(m/z, charge)",
    fill=PALE_BLUE_2, fontsize=9)

box(11.0, 16.3, 4.2, 0.7,
    "Peptide sequence   $[B,\\,42]$   (AA token ids)",
    fill=PALE_BLUE_2, fontsize=10)

# ---- first-level embeddings ------------------------------------------------
box(3.0, 15.0, 4.4, 1.0,
    "MultiScalePeakEmbedding\n"
    "$\\mathrm{sin/cos}(\\omega \\cdot m\\!/\\!z)\\;\\rightarrow\\;\\mathrm{MLP}\\;\\rightarrow\\;[\\,\\cdot\\,,\\,I]\\;\\rightarrow\\;\\mathrm{head}$\n"
    "$[B,\\,500,\\,d]$",
    fontsize=9)

box(6.3, 15.0, 2.5, 1.0,
    "Precursor encoder\n"
    "shared peak enc. on $(m\\!/\\!z,1)$\n"
    "$\\oplus$ charge $\\mathrm{Emb}(5,d)$\n"
    "$\\mathrm{Linear}(2d\\!\\to\\!d)$",
    fontsize=8)

box(11.0, 15.0, 4.4, 1.0,
    "AA Embedding\n"
    "$\\mathrm{Embed}(26,\\,d,\\ \\mathrm{padding\\_idx}=0)$\n"
    "$[B,\\,42,\\,d]$",
    fontsize=9)

# ---- CLS / precursor merge / pos enc ---------------------------------------
box(3.0, 13.4, 4.8, 1.0,
    "CLS $\\leftarrow$ CLS$_{\\text{learned}}$ $+$ precursor embedding\n"
    "Concat $[\\,\\text{CLS};\\,\\text{peaks}\\,]$  $\\rightarrow\\;[B,\\,501,\\,d]$\n"
    "Key-padding mask from all-zero peak rows",
    fontsize=9)

box(11.0, 13.4, 4.8, 1.0,
    "Prepend learned CLS token\n"
    "Sinusoidal positional encoding\n"
    "$[B,\\,43,\\,d]$,  pad mask where $\\mathrm{token}=0$",
    fontsize=9)

# ---- transformer stack -----------------------------------------------------
stack_box(3.0, 11.55, 4.4, 1.25,
          "Transformer Encoder  $\\times\\,N_{\\mathrm{layers}}$\n"
          "Pre-LN  $\\rightarrow$  Multi-Head Self-Attention\n"
          "$\\rightarrow$  FFN (Linear–GELU–Dropout–Linear)\n"
          "with key-padding mask",
          fontsize=9, depth=2)

stack_box(11.0, 11.55, 4.4, 1.25,
          "Transformer Encoder  $\\times\\,N_{\\mathrm{layers}}$\n"
          "Pre-LN  $\\rightarrow$  Multi-Head Self-Attention\n"
          "$\\rightarrow$  FFN (Linear–GELU–Dropout–Linear)\n"
          "with key-padding mask",
          fontsize=9, depth=2)

# ---- final LN on CLS -------------------------------------------------------
box(3.0, 9.9, 3.6, 0.55, "LayerNorm on CLS token  $x[:,0,:]$", fontsize=9)
box(11.0, 9.9, 3.6, 0.55, "LayerNorm on CLS token  $x[:,0,:]$", fontsize=9)

# ---- projection head -------------------------------------------------------
box(3.0, 8.9, 4.2, 0.9,
    "Projection Head\n"
    "Linear($d$, $2d$)  $\\rightarrow$  GELU  $\\rightarrow$  LN  $\\rightarrow$  Dropout  $\\rightarrow$  Linear($2d$, $D$)",
    fontsize=9)
box(11.0, 8.9, 4.2, 0.9,
    "Projection Head\n"
    "Linear($d$, $2d$)  $\\rightarrow$  GELU  $\\rightarrow$  LN  $\\rightarrow$  Dropout  $\\rightarrow$  Linear($2d$, $D$)",
    fontsize=9)

# ---- L2 normalize ----------------------------------------------------------
box(3.0, 7.95, 3.0, 0.5, "$\\ell_{2}$ normalize", fontsize=9)
box(11.0, 7.95, 3.0, 0.5, "$\\ell_{2}$ normalize", fontsize=9)

# ---- output embeddings -----------------------------------------------------
box(3.0, 7.05, 2.8, 0.55,
    "$\\mathbf{z}_{\\mathrm{spec}} \\in \\mathbb{R}^{B \\times D}$",
    fill=PALE_BLUE_2, fontsize=11, weight="bold")
box(11.0, 7.05, 2.8, 0.55,
    "$\\mathbf{z}_{\\mathrm{pep}}\\; \\in \\mathbb{R}^{B \\times D}$",
    fill=PALE_BLUE_2, fontsize=11, weight="bold")

# ---- similarity ------------------------------------------------------------
box(7.0, 5.55, 8.2, 0.95,
    "Similarity matrix     "
    "$S \\;=\\; \\mathbf{z}_{\\mathrm{spec}}\\,\\mathbf{z}_{\\mathrm{pep}}^{\\top} \\,/\\, \\tau$,"
    "      $\\tau = \\exp(\\log\\tau)$  clamped to $[0.04,\\,0.5]$",
    fontsize=10)

# ---- loss -----------------------------------------------------------------
box(7.0, 4.15, 8.2, 1.05,
    "Symmetric cross-entropy  (label smoothing $= 0.1$)\n"
    "$\\mathcal{L} \\;=\\; \\frac{1}{2}\\left[\\mathrm{CE}(S,\\,I) \\;+\\; \\mathrm{CE}(S^{\\top},\\,I)\\right]$",
    fill=PALE_BLUE_2, fontsize=10.5, weight="bold")

# ---- footer / config notes -------------------------------------------------
ax.text(FIG_W / 2, 2.5,
        "Shared hyper-parameters:  "
        "$d=d_{\\mathrm{model}}$,   $d_{\\mathrm{ff}}$,   $N_{\\mathrm{heads}}$,   "
        "$N_{\\mathrm{layers}}$,   embedding dim $D$,   dropout $p$",
        ha="center", va="center", fontsize=10, style="italic",
        family="serif", color=TEXT)
ax.text(FIG_W / 2, 2.0,
        "Padding: spectrum peaks padded to 500 with zeros;  peptides padded to 42 with token id 0.",
        ha="center", va="center", fontsize=9.5,
        family="serif", color=TEXT)
ax.text(FIG_W / 2, 1.55,
        "Only the final CLS representation is projected; all other tokens are discarded after the transformer stack.",
        ha="center", va="center", fontsize=9.5,
        family="serif", color=TEXT)

# ===========================================================================
# arrows
# ===========================================================================

# -- spectrum main column (x=3) ---------------------------------------------
arrow(3.0, 15.95, 3.0, 15.51)         # input -> peak embedding
arrow(3.0, 14.50, 3.0, 13.91)         # peak emb -> CLS merge
arrow(3.0, 12.89, 3.0, 12.25)         # CLS merge -> transformer
arrow(3.0, 10.85, 3.0, 10.18)         # transformer -> LN
arrow(3.0, 9.62, 3.0, 9.36)           # LN -> projection
arrow(3.0, 8.44, 3.0, 8.21)           # projection -> L2 normalize
arrow(3.0, 7.69, 3.0, 7.33)           # L2 -> z_spec

# -- precursor subbranch (x=6.3) -> merges to CLS row -----------------------
arrow(6.3, 15.95, 6.3, 15.51)         # precursor input -> precursor encoder
# precursor encoder -> CLS merge box (curve into main spectrum column)
arrow(6.3, 14.50, 4.2, 13.92, style="-|>")

# -- peptide column (x=11) --------------------------------------------------
arrow(11.0, 15.95, 11.0, 15.51)
arrow(11.0, 14.50, 11.0, 13.91)
arrow(11.0, 12.89, 11.0, 12.25)
arrow(11.0, 10.85, 11.0, 10.18)
arrow(11.0, 9.62, 11.0, 9.36)
arrow(11.0, 8.44, 11.0, 8.21)
arrow(11.0, 7.69, 11.0, 7.33)

# -- embeddings into similarity --------------------------------------------
arrow(3.0, 6.77, 5.5, 6.03)
arrow(11.0, 6.77, 8.5, 6.03)

# -- similarity into loss ---------------------------------------------------
arrow(7.0, 5.07, 7.0, 4.68)

# ---- save -----------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "dual_encoder_architecture.png")
plt.savefig(out, dpi=240, facecolor=WHITE, bbox_inches="tight", pad_inches=0.25)
print(f"Saved: {out}")
