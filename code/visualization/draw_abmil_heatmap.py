import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import warnings

# ────────────────────────────────────────────────────────────────────────────
# Fix Windows OpenSlide issue by explicitly adding DLL directory BEFORE import
# ────────────────────────────────────────────────────────────────────────────
DLL_PATH = r"D:\Professionalsoftware\openslide-bin-4.0.0.6-windows-x64\bin"
if os.path.exists(DLL_PATH):
    os.add_dll_directory(DLL_PATH)
import openslide

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
# Model settings
MODEL_NAME    = "uni_v2"
FEAT_SUBDIR   = "features_uni_v2"
D_FEAT        = 1536          # matches UNI v2
N_CLASSES     = 2             # benchmark uses binary D3/M3 classification

# Paths
BASE_DIR      = r"D:\BaiduSyncdisk\work\pycharm\UVM_ly"
SVS_DIR       = r"F:\UVM\SVS"
FEAT_DIR      = os.path.join(r"D:\Datas of lab\UVM\Trident\20x_256px_0px_overlap", FEAT_SUBDIR)
CLINICAL_PATH = r"D:\Datas of lab\UVM\临床表\临床信息表.csv"

# Pick the best fold model from the benchmark output
BEST_MODEL_PATH = os.path.join(BASE_DIR, "_3_predictmodel", "benchmark_output",
                               "20260331_171049", MODEL_NAME, "fold_1", "best_model.pth")
OUT_DIR       = os.path.join(BASE_DIR, "_3_predictmodel", "heatmap_output")
os.makedirs(OUT_DIR, exist_ok=True)

# Heatmap rendering settings
THUMBNAIL_LEVEL = 2       # Usually level 2 or 3 in SVS is good for thumbnails
PATCH_SIZE      = 256     # The size patches were extracted at (level 0)
RENDER_MODE     = 'multiply' # 'overlay' or 'multiply'

# ────────────────────────────────────────────────────────────────────────────
# Model — import the ACTUAL architecture used by the benchmark
# ────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(BASE_DIR, "_3_predictmodel"))
from architecture.abmil_cls import ABMIL_Cls

# ────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ────────────────────────────────────────────────────────────────────────────
def create_custom_cmap():
    """Attention colormap (low=purple/black, high=red/orange)."""
    colors = [
        (0.00, (0.001, 0.001, 0.015)),
        (0.25, (0.30, 0.08, 0.45)),
        (0.50, (0.98, 0.75, 0.60)),
        (0.75, (0.85, 0.30, 0.20)),
        (1.00, (0.60, 0.10, 0.05)),
    ]
    return LinearSegmentedColormap.from_list('custom_magma', colors, N=256)

def normalize_scores(scores, clip_pct=(1, 99)):
    low = np.percentile(scores, clip_pct[0])
    high = np.percentile(scores, clip_pct[1])
    if high > low:
        norm = (np.clip(scores, low, high) - low) / (high - low)
    else:
        norm = np.full_like(scores, 0.5)
    return norm

def render_multiply(patch_array, score_norm, cmap):
    color_norm = np.array(cmap(score_norm)[:3])
    patch_float = patch_array.astype(np.float32) / 255.0
    color_adjusted = color_norm * 0.7 + 0.3
    blended = patch_float * color_adjusted
    return np.clip(blended * 255, 0, 255).astype(np.uint8)

def load_svs_thumbnail(svs_path, level=THUMBNAIL_LEVEL):
    slide = openslide.OpenSlide(svs_path)
    # If the requested level doesn't exist, use the lowest resolution available
    if level >= slide.level_count:
        level = slide.level_count - 1

    dimensions = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    # Read thumbnail image
    thumb = slide.read_region((0,0), level, dimensions).convert("RGB")
    slide.close()
    return np.array(thumb), downsample

# ────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ────────────────────────────────────────────────────────────────────────────
def generate_heatmap(slide_id, model, device):
    print(f"\nProcessing {slide_id}...")

    # 1. Locate files
    h5_path = os.path.join(FEAT_DIR, f"{slide_id}.h5")

    # Locate SVS (might be nested in subfolders like F:/UVM/SVS/<uuid>/TCGA-XXX.svs)
    svs_path = None
    for root, dirs, files in os.walk(SVS_DIR):
        for f in files:
            if f.startswith(slide_id) and f.endswith(".svs"):
                svs_path = os.path.join(root, f)
                break
        if svs_path:
            break

    if not os.path.exists(h5_path):
        print(f"Error: Missing feature file {h5_path}")
        return
    if not svs_path or not os.path.exists(svs_path):
        print(f"Error: Missing SVS file for {slide_id}")
        return

    # 2. Extract features and coordinates
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        feat_key = 'features' if 'features' in keys else ('feats' if 'feats' in keys else None)
        if feat_key is None:
            feat_key = [k for k in keys if k != 'coords'][0]

        feats = torch.from_numpy(f[feat_key][:]).float().to(device)
        coords = f['coords'][:] # [N, 2] Level 0 (X,Y) coordinates

    # 3. Model Forward Pass -> Attention
    model.eval()
    with torch.no_grad():
        logits, A_soft = model(feats)
        pred_class = "?"
        attn_scores = A_soft.cpu().numpy()[0]

    print(f"  Attention computed")

    # Normalize attention
    norm_attn = normalize_scores(attn_scores)

    # 4. Load Thumbnail
    thumb_arr, downsample = load_svs_thumbnail(svs_path, level=THUMBNAIL_LEVEL)
    print(f"  Thumbnail size: {thumb_arr.shape[:2]}, Downsample: {downsample:.2f}x")

    # 5. Render Heatmap on Thumbnail
    heatmap = thumb_arr.copy()
    cmap = create_custom_cmap()

    # Size of a patch on the thumbnail
    patch_size_thumb = max(1, int(PATCH_SIZE / downsample))

    placed = 0
    for i in range(len(coords)):
        x_level0, y_level0 = coords[i]

        # Convert to thumbnail coordinates
        x_thumb = int(x_level0 / downsample)
        y_thumb = int(y_level0 / downsample)

        # Boundary check
        if (x_thumb >= 0 and y_thumb >= 0 and
            x_thumb + patch_size_thumb <= heatmap.shape[1] and
            y_thumb + patch_size_thumb <= heatmap.shape[0]):

            # Get background patch from thumbnail
            bg_patch = thumb_arr[y_thumb:y_thumb+patch_size_thumb, x_thumb:x_thumb+patch_size_thumb]

            # Apply attention color via multiply
            colored_patch = render_multiply(bg_patch, norm_attn[i], cmap)
            heatmap[y_thumb:y_thumb+patch_size_thumb, x_thumb:x_thumb+patch_size_thumb] = colored_patch
            placed += 1

    print(f"  Rendered {placed}/{len(coords)} patches on thumbnail")

    # 6. Plot and Save
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(heatmap)
    ax.axis('off')
    ax.set_title(f'ABMIL Attention Heatmap - {slide_id}\nModel: {MODEL_NAME} | Pred: Class {pred_class}',
                 fontsize=16, pad=20)

    # Add colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Score (Normalized)', rotation=270, labelpad=25, fontsize=12)

    out_file = os.path.join(OUT_DIR, f"{slide_id}_{MODEL_NAME}_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap to {out_file}")


def main():
    print(f"Initializing ABMIL Heatmap Generator for {MODEL_NAME}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model — use the same architecture as the benchmark
    D_INNER = D_FEAT // 2
    model = ABMIL_Cls(D_feat=D_FEAT, D_inner=D_INNER, n_classes=N_CLASSES)

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading weights from {BEST_MODEL_PATH}")
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
    else:
        raise FileNotFoundError(f"Weights not found at {BEST_MODEL_PATH}. Cannot generate heatmap without trained model.")

    model.to(device)

    # Select some slides to generate heatmaps for
    # (Just grab the first 3 feature files as a test)
    feat_files = [f for f in os.listdir(FEAT_DIR) if f.endswith('.h5')][:3]
    slide_ids = [f.replace('.h5', '') for f in feat_files]

    for sid in slide_ids:
        generate_heatmap(sid, model, device)

if __name__ == "__main__":
    main()
