#!/usr/bin/env python3
"""
Save individual image panels from the leaf-autoencoder preprocessing pipeline
as PNG files in figures/supplemental/.

Requires: numpy, matplotlib, Pillow, scipy, scikit-learn
Run from the project root:
    python figures/supplemental/make_preprocessing_workflow.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
STEM  = "1605_LeafPhotoA_2025-09-09 09_08_08.350-05_00"
RAW   = os.path.join(BASE, f"data/ne2025/device6/{STEM}.jpg")
MASK  = os.path.join(BASE, f"data/processed/ne2025/device6/masks/{STEM}.png")
CDIR  = os.path.join(BASE, "data/processed/ne2025/device6/cropped")
MCDIR = os.path.join(BASE, "data/processed/ne2025/device6/masks_cropped")
OUTDIR = os.path.join(BASE, "figures/supplemental")

# ─── Load raw image and binary mask ──────────────────────────────────────────
img_full  = np.array(Image.open(RAW))   # (4080, 3060, 3) uint8
mask_full = np.array(Image.open(MASK))  # (4080, 3060)    uint8, 0 or 255
H, W = img_full.shape[:2]

SC  = 0.08                                  # display downsample factor
h_s = int(H * SC);  w_s = int(W * SC)
img_s  = np.array(Image.fromarray(img_full ).resize((w_s, h_s), Image.LANCZOS))
mask_s = np.array(Image.fromarray(mask_full).resize((w_s, h_s), Image.NEAREST))


# ─── CV segmentation helpers ──────────────────────────────────────────────────
def flood_region(arr, sy, sx, tol=50):
    """Flood-fill region: pixels connected to (sy, sx) within per-channel tol."""
    seed  = arr[sy, sx].astype(np.int32)
    diff  = np.max(np.abs(arr.astype(np.int32) - seed), axis=2)
    inrng = diff <= tol
    lab, _ = ndimage.label(inrng)
    lbl = lab[sy, sx]
    return (lab == lbl) if lbl > 0 else np.zeros(arr.shape[:2], bool)


# Seed positions at display scale
s1y, s1x = int(750 * SC), w_s // 2          # top-center seed
s2y, s2x = h_s - max(1, int(20 * SC)), w_s // 2  # bottom-center seed

# Step 1: two flood fills remove background
fill1 = flood_region(img_s, s1y, s1x)
work1 = img_s.copy();  work1[fill1] = 0

fill2 = flood_region(work1, s2y, s2x)
work2 = work1.copy();  work2[fill2] = 0

# Step 2: largest connected component touching both left AND right edges
fg  = np.any(work2 != 0, axis=2).astype(np.uint8)
lab, n = ndimage.label(fg)
best_lbl, best_area = None, 0
for lbl in range(1, n + 1):
    comp = lab == lbl
    if comp[:, 0].any() and comp[:, -1].any():
        area = int(comp.sum())
        if area > best_area:
            best_area, best_lbl = area, lbl
comp_mask = (lab == best_lbl) if best_lbl else fg.astype(bool)

# Step 3: trim left / right edges; keep largest remaining component
tl, tr = int(300 * SC), int(100 * SC)
trimmed = comp_mask.copy()
trimmed[:, :tl] = False
trimmed[:, w_s - tr:] = False
lab2, n2 = ndimage.label(trimmed.astype(np.uint8))
best2, best2_area = None, 0
for lbl in range(1, n2 + 1):
    area = int((lab2 == lbl).sum())
    if area > best2_area:
        best2_area, best2 = area, lbl
final_mask_s = (lab2 == best2).astype(np.uint8) if best2 else trimmed.astype(np.uint8)
bw_s = (final_mask_s * 255).astype(np.uint8)


# ─── PCA + crop rectangle calculation (full resolution) ──────────────────────
yc, xc = np.where(mask_full > 0)
pts = np.column_stack((xc, yc)).astype(np.float64)
pca_model = PCA(n_components=2).fit(pts)
pa   = pca_model.components_[0]   # principal axis
perp = pca_model.components_[1]   # perpendicular axis
cx_f, cy_f = pts[:, 0].mean(), pts[:, 1].mean()

proj  = np.dot(pts - [cx_f, cy_f], pa)
pperp = np.dot(pts - [cx_f, cy_f], perp)
# Ensure pa has greater extent than perp
if (pperp.max() - pperp.min()) > (proj.max() - proj.min()):
    pa, perp = perp, pa
    proj, pperp = pperp, proj

min_p, max_p = proj.min(), proj.max()
start_f = np.array([cx_f, cy_f]) + min_p * pa

X_DIM, Y_DIM, STEP = 1000, 2000, 500
crop_corners_s = []   # corners scaled for display
d = 0.0
while d + X_DIM <= max_p - min_p:
    wc_ax = start_f + (d + X_DIM / 2) * pa
    in_win = (proj >= min_p + d) & (proj <= min_p + d + X_DIM)
    if in_win.sum() == 0:
        d += STEP; continue
    mean_perp_offset = pperp[in_win].mean()
    wc = wc_ax + mean_perp_offset * perp
    hx, hy = X_DIM / 2, Y_DIM / 2
    corners = np.array([
        wc + (-hx) * pa + (-hy) * perp,
        wc + ( hx) * pa + (-hy) * perp,
        wc + ( hx) * pa + ( hy) * perp,
        wc + (-hx) * pa + ( hy) * perp,
    ])  # shape (4, 2), [x, y] in full-res pixels
    if (corners[:, 0].min() >= 0 and corners[:, 0].max() < W and
            corners[:, 1].min() >= 0 and corners[:, 1].max() < H):
        crop_corners_s.append(corners * SC)   # scale to display pixels
    d += STEP


# ─── Cropped image data ───────────────────────────────────────────────────────
crop_files = sorted(f for f in os.listdir(CDIR) if f.startswith(STEM))[:3]
crops = [np.array(Image.open(os.path.join(CDIR, f))) for f in crop_files]

masked_crops = []
for f in crop_files:
    p = os.path.join(MCDIR, f)
    mc = np.array(Image.open(p)) if os.path.exists(p) else np.zeros(crops[0].shape[:2], np.uint8)
    masked_crops.append(mc[..., None])

# Masked full image at display scale
mask_bool_s  = mask_s > 0
masked_img_s = img_s.copy()
masked_img_s[~mask_bool_s] = 0


# ─── Simulated SAM3 output ────────────────────────────────────────────────────
yc_s, xc_s = np.where(mask_bool_s)
pp_vals = (xc_s - w_s / 2) * perp[0] + (yc_s - h_s / 2) * perp[1]
med_pp  = float(np.median(pp_vals))
seg1 = np.zeros((h_s, w_s), bool)
seg2 = np.zeros((h_s, w_s), bool)
for yy, xx, pp in zip(yc_s, xc_s, pp_vals):
    (seg1 if pp <= med_pp else seg2)[yy, xx] = True

TEAL  = np.array([20, 185, 175], np.uint8)
PINK  = np.array([185, 65,  145], np.uint8)
alpha = 0.55
sam3_vis = img_s.copy()
sam3_vis[seg1] = (alpha * TEAL + (1 - alpha) * img_s[seg1]).astype(np.uint8)
sam3_vis[seg2] = (alpha * PINK + (1 - alpha) * img_s[seg2]).astype(np.uint8)


# ─── Panel save helper ────────────────────────────────────────────────────────
C_PA   = '#d94f00'
C_PERP = '#0070c0'
C_CROP = '#ffe000'

def save_panel(img_data, filename, cmap=None, overlays_fn=None, dpi=300):
    h, w = img_data.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi))
    fig.subplots_adjust(0, 0, 1, 1)
    kw = dict(aspect='auto', interpolation='bilinear')
    if cmap:
        kw['cmap'] = cmap
    ax.imshow(img_data, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if overlays_fn:
        overlays_fn(ax)
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    print(f"Saved {path}")
    plt.close(fig)


# ─── Section 1 — OpenCV segmentation ─────────────────────────────────────────
def seed_dots(ax):
    ax.plot(s1x, s1y, 'o', color='cyan', ms=5, mew=0, zorder=5)
    ax.plot(s2x, s2y, 'o', color='cyan', ms=5, mew=0, zorder=5)

save_panel(img_s,                                   'step1_original.png',           overlays_fn=seed_dots)
save_panel(work2,                                   'step1_flood_fill.png',         overlays_fn=seed_dots)
save_panel(comp_mask.astype(np.uint8) * 255,        'step1_largest_component.png',  cmap='gray')
save_panel(bw_s,                                    'step1_bw_mask.png',            cmap='gray')


# ─── Section 1b — SAM3 ───────────────────────────────────────────────────────
save_panel(img_s,    'step1b_original.png')
save_panel(sam3_vis, 'step1b_sam3_segments.png')


# ─── Section 2 — Align and crop ──────────────────────────────────────────────
arrow_len_px = w_s * 0.36
cx_s = cx_f * SC
cy_s = cy_f * SC

def pca_arrows(ax):
    for axis_vec, col in [(pa, C_PA), (perp, C_PERP)]:
        dx, dy = axis_vec[0] * arrow_len_px, axis_vec[1] * arrow_len_px
        ax.annotate('', xy=(cx_s + dx, cy_s + dy), xytext=(cx_s - dx, cy_s - dy),
                    arrowprops=dict(arrowstyle='<->', color=col, lw=1.8, mutation_scale=7))

def crop_rects(ax):
    cmap = plt.cm.tab10
    n = max(len(crop_corners_s), 1)
    for i, corners in enumerate(crop_corners_s):
        color = cmap(i / n)
        poly = plt.Polygon(corners, closed=True, fill=False,
                           edgecolor=color, linewidth=0.9)
        ax.add_patch(poly)

save_panel(img_s,        'step2_rgb_pca.png',          overlays_fn=pca_arrows)
save_panel(img_s,        'step2_rgb_crop_windows.png', overlays_fn=crop_rects)
save_panel(mask_s, 'step2_masked_pca.png',       overlays_fn=pca_arrows, cmap='grey')
save_panel(mask_s, 'step2_masked_crop_windows.png', overlays_fn=crop_rects, cmap='grey')

for i, crop in enumerate(crops):
    save_panel(crop, f'step2_rgb_crop_{i}.png')
for i, mcrop in enumerate(masked_crops):
    save_panel(mcrop, f'step2_masked_crop_{i}.png')


# ─── Section 4 — Apply masks ─────────────────────────────────────────────────
save_panel(crops[0],        'step4_without_mask.png')
save_panel(masked_crops[0], 'step4_with_mask.png')
