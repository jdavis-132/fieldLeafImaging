"""
Plot gene models for Chr06:58560985-61826680 from Sorghum bicolor GFF3,
with a GWAS -log10(P) scatter panel above.
Saves as SVG, 6.5 x 3.0 inches, Helvetica 9pt font.
"""

import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from matplotlib import font_manager
import numpy as np
import pandas as pd

for _ttc in [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]:
    import os
    if os.path.exists(_ttc):
        font_manager.fontManager.addfont(_ttc)

# ── parameters ────────────────────────────────────────────────────────────────
GFF  = "data/genotype/Sbicolor_730_v5.1.gene.gff3"
GWAS = "output/mlm_20260515/GWAS_embedding_mean_768_all_results.csv"
CHROM = "Chr06"
START = 58483464
END = 58683464
# END   = 61826680
OUT   = "figures/main/figure3/gene_models.svg"

FIG_W, FIG_H = 6.5, 1.5
FONT_SIZE = 9

# track geometry
GENE_HEIGHT   = 0.35   # height of exon/CDS boxes
UTR_HEIGHT    = 0.20   # height of UTR boxes (narrower)
TRACK_SPACING = 0.55   # vertical distance between track centres
GENE_PAD      = 5000   # bp padding between genes in same track (avoids label clash)

# colours
COLOR_CDS  = "#A6CEE3FF"
COLOR_UTR  = "#A6CEE3FF"
COLOR_LINE = "#A6CEE3FF"

HIGHLIGHT_GENE  = "Sobic.006G226800"
COLOR_HI_CDS    = "#6A3D9AFF"
COLOR_HI_UTR    = "#6A3D9AFF"
COLOR_HI_LINE   = "#6A3D9AFF"

FONT_FAMILY = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica Neue', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams["font.size"]   = FONT_SIZE
matplotlib.rcParams["svg.fonttype"] = "none"

# ── parse GFF3 ────────────────────────────────────────────────────────────────
def parse_attrs(attr_str):
    attrs = {}
    for field in attr_str.strip().split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            attrs[k] = v
    return attrs

genes   = {}   # gene_id -> {start, end, strand, name}
mrnas   = {}   # mrna_id -> {gene_id, start, end, strand, longest}
features = {}  # mrna_id -> list of (type, start, end)

with open(GFF) as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 9:
            continue
        chrom, _, ftype, s, e, _, strand, _, attr_str = cols
        if chrom != CHROM:
            continue
        s, e = int(s), int(e)
        # keep only records that overlap the window
        if s > END or e < START:
            continue

        attrs = parse_attrs(attr_str)

        if ftype == "gene":
            gid = attrs.get("ID", "")
            genes[gid] = dict(start=s, end=e, strand=strand,
                              name=attrs.get("Name", gid))

        elif ftype == "mRNA":
            mid = attrs.get("ID", "")
            mrnas[mid] = dict(
                gene_id  = attrs.get("Parent", ""),
                start    = s, end=e, strand=strand,
                longest  = attrs.get("longest", "0") == "1",
            )
            features[mid] = []

        elif ftype in ("CDS", "five_prime_UTR", "three_prime_UTR"):
            parent = attrs.get("Parent", "")
            if parent in features:
                features[parent].append((ftype, s, e))

# ── select canonical (longest) transcript per gene ────────────────────────────
gene_transcripts = {}   # gene_id -> mrna_id
for mid, m in mrnas.items():
    gid = m["gene_id"]
    if gid not in gene_transcripts:
        gene_transcripts[gid] = mid
    elif m["longest"] and not mrnas[gene_transcripts[gid]]["longest"]:
        gene_transcripts[gid] = mid

# build list of gene records for plotting
records = []
for gid, g in genes.items():
    mid = gene_transcripts.get(gid)
    feats = features.get(mid, []) if mid else []
    records.append(dict(
        gid    = gid,
        name   = g["name"],
        start  = g["start"],
        end    = g["end"],
        strand = g["strand"],
        feats  = feats,
    ))

records.sort(key=lambda r: r["start"])

# ── greedy track packing ───────────────────────────────────────────────────────
tracks = []   # list of lists of records
track_ends = []  # rightmost end bp in each track

for rec in records:
    placed = False
    for i, tend in enumerate(track_ends):
        if rec["start"] - tend >= GENE_PAD:
            tracks[i].append(rec)
            track_ends[i] = rec["end"]
            placed = True
            break
    if not placed:
        tracks.append([rec])
        track_ends.append(rec["end"])

# ── load GWAS data ─────────────────────────────────────────────────────────────
gwas = pd.read_csv(GWAS)
gwas = gwas[(gwas["CHROM"] == 6) & (gwas["POS"] >= START) & (gwas["POS"] <= END)].copy()
gwas["neglog10p"] = -np.log10(gwas["MLM_P"])

# ── filter genes: keep only those with at least one GWAS position (-log10(P) > 4) ──
sig_pos = gwas.loc[gwas["neglog10p"] > 6, "POS"].values

def gene_has_sig_hit(rec):
    return np.any((sig_pos >= rec["start"]) & (sig_pos <= rec["end"]))

records = [r for r in records if gene_has_sig_hit(r)]

# re-pack into tracks with filtered records
tracks = []
track_ends = []
for rec in records:
    placed = False
    for i, tend in enumerate(track_ends):
        if rec["start"] - tend >= GENE_PAD:
            tracks[i].append(rec)
            track_ends[i] = rec["end"]
            placed = True
            break
    if not placed:
        tracks.append([rec])
        track_ends.append(rec["end"])

n_tracks = len(tracks)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_gwas, ax) = plt.subplots(
    2, 1, figsize=(FIG_W, FIG_H),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    layout="constrained",
)

total_bp = END - START
arrow_size = total_bp * 0.003   # arrowhead width in bp

def draw_gene(ax, rec, y_centre):
    s    = rec["start"]
    e    = rec["end"]
    strd = rec["strand"]
    feats = rec["feats"]

    highlight = rec["name"] == HIGHLIGHT_GENE
    c_line = COLOR_HI_LINE if highlight else COLOR_LINE
    c_cds  = COLOR_HI_CDS  if highlight else COLOR_CDS
    c_utr  = COLOR_HI_UTR  if highlight else COLOR_UTR

    # clip to window
    s_clip = max(s, START)
    e_clip = min(e, END)

    # backbone (intron) line
    ax.plot([s_clip, e_clip], [y_centre, y_centre],
            color=c_line, linewidth=1, solid_capstyle="butt", zorder=1)

    # draw direction arrow on backbone midpoint
    mid = (s_clip + e_clip) / 2
    dx  = arrow_size * (1 if strd == "+" else -1)
    ax.annotate("",
                xy=(mid + dx, y_centre),
                xytext=(mid, y_centre),
                arrowprops=dict(arrowstyle="-|>", color=c_line,
                                lw=0.5, mutation_scale=4),
                zorder=2)

    # draw features
    for ftype, fs, fe in feats:
        fs_c = max(fs, START)
        fe_c = min(fe, END)
        if fs_c >= fe_c:
            continue
        if ftype == "CDS":
            h = GENE_HEIGHT
            color = c_cds
        else:
            h = UTR_HEIGHT
            color = c_utr
        rect = mpatches.FancyBboxPatch(
            (fs_c, y_centre - h / 2), fe_c - fs_c, h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="none",
            linewidth=0, zorder=3,
        )
        ax.add_patch(rect)

# y positions: track 0 at top, increasing downward
for ti, track in enumerate(tracks):
    y = -(ti * TRACK_SPACING)
    for rec in track:
        draw_gene(ax, rec, y)

# ── GWAS panel ────────────────────────────────────────────────────────────────
ax_gwas.scatter(gwas["POS"], gwas["neglog10p"],
                s=2, color="#333333", linewidths=0, rasterized=True)
ax_gwas.set_xlim(START, END)
ax_gwas.set_ylabel("-log\u2081\u2080(p)", fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
ax_gwas.set_ylim(0, 8)
ax_gwas.set_yticks(ax_gwas.get_yticks())
ax_gwas.tick_params(axis="y", length=2, width=0.5, labelsize=FONT_SIZE)
for spine in ["top", "right"]:
    ax_gwas.spines[spine].set_visible(False)
ax_gwas.spines["left"].set_linewidth(0.5)
ax_gwas.spines["bottom"].set_visible(False)
ax_gwas.tick_params(axis="x", bottom=False)

# ── axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(START, END)

y_bottom = -(n_tracks - 1) * TRACK_SPACING - TRACK_SPACING * 0.6
y_top    = TRACK_SPACING * 0.6
ax.set_ylim(y_bottom, y_top)

# x-axis: Mb ticks
mb_step = 0.5e5
tick_vals = np.arange(
    np.ceil(START / mb_step) * mb_step,
    END + 1,
    mb_step,
)
ax.set_xticks(tick_vals)
ax.set_xticklabels([f"{v/1e6:.2f}" for v in tick_vals],
                   fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
ax.set_xlabel("Chromosome 6 position (Mb)", fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)

ax.set_yticks([])
for spine in ["top", "left", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(axis="x", length=2, width=0.5, labelsize=FONT_SIZE)


fig.savefig(OUT, format="svg", dpi=300, bbox_inches="tight")
print(f"Saved → {OUT}  ({n_tracks} tracks, {len(records)} genes)")
