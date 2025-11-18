"""
Create a professional architectural diagram for the Disease-Aware U-Net Autoencoder.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set up the figure with 16:9 aspect ratio
fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color scheme
colors = {
    'input': '#4A90E2',
    'conv': '#7B68EE',
    'pool': '#FF6B6B',
    'bottleneck': '#E74C3C',
    'attention': '#F39C12',
    'upconv': '#2ECC71',
    'concat': '#9B59B6',
    'output': '#1ABC9C',
    'skip': '#E67E22'
}

def draw_block(x, y, width, height, color, label, sublabel='', dimensions='', alpha=0.9):
    """Draw a colored block with labels."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor='black',
        linewidth=2, alpha=alpha, zorder=3
    )
    ax.add_patch(box)

    # Main label
    ax.text(x, y + 0.3, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)

    # Sublabel
    if sublabel:
        ax.text(x, y - 0.4, sublabel, ha='center', va='center',
                fontsize=8, color='white', zorder=4, style='italic')

    # Dimensions (above block)
    if dimensions:
        ax.text(x, y + height/2 + 1.2, dimensions, ha='center', va='bottom',
                fontsize=8, fontweight='bold', family='monospace',
                color='#2C3E50', zorder=4)

def draw_arrow(x1, y1, x2, y2, color='gray', style='solid', width=2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=20,
        linewidth=width, color=color,
        linestyle=style, zorder=2
    )
    ax.add_patch(arrow)

def draw_curved_skip(x1, y1, x2, y2, color=colors['skip']):
    """Draw curved skip connection."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=15,
        linewidth=2.5, color=color,
        linestyle='dashed',
        connectionstyle=f"arc3,rad=.5", zorder=1
    )
    ax.add_patch(arrow)

# Title
ax.text(50, 97, 'Disease-Aware U-Net Autoencoder Architecture',
        ha='center', va='top', fontsize=24, fontweight='bold', color='#2C3E50')
ax.text(50, 93.5, 'Input: LAB + Mask (4ch) → Encoder → Attention Bottleneck → Decoder → Output: LAB (3ch)',
        ha='center', va='top', fontsize=12, color='#34495E')

# Layout parameters
encoder_x = 12
decoder_x = 72
start_y = 85
y_step = 11

# Input
draw_block(encoder_x, start_y, 6, 4, colors['input'], 'Input', 'LAB+Mask', '(B,4,224,224)')

# Encoder stages
encoder_stages = [
    {'ch': 64, 'size': 224, 'label': 'Conv Block 1'},
    {'ch': 128, 'size': 112, 'label': 'Conv Block 2'},
    {'ch': 256, 'size': 56, 'label': 'Conv Block 3'},
    {'ch': 512, 'size': 28, 'label': 'Conv Block 4'}
]

current_y = start_y
skip_positions = []

for i, stage in enumerate(encoder_stages):
    # Arrow down
    draw_arrow(encoder_x, current_y - 2, encoder_x, current_y - 2 - y_step + 2, 'black', 'solid')
    current_y -= y_step

    # Conv block
    dim_str = f"(B,{stage['ch']},{stage['size']},{stage['size']})"
    draw_block(encoder_x, current_y, 6, 4, colors['conv'],
               stage['label'], '2×Conv3×3', dim_str)

    skip_positions.append((encoder_x + 3, current_y))

    # MaxPool (except last)
    if i < len(encoder_stages) - 1:
        draw_arrow(encoder_x, current_y - 2, encoder_x, current_y - 4, 'black', 'solid')
        current_y -= 4
        draw_block(encoder_x, current_y, 6, 2, colors['pool'],
                   'MaxPool', '2×2, s=2', '')
        current_y -= 1

# Bottleneck
draw_arrow(encoder_x, current_y - 2, encoder_x, current_y - 5, 'black', 'solid')
current_y -= 5
bottleneck_y = current_y

draw_block(encoder_x, bottleneck_y, 6, 4, colors['bottleneck'],
           'Bottleneck', '2×Conv3×3', '(B,1024,14,14)')

# Attention
draw_arrow(encoder_x, bottleneck_y - 2, encoder_x, bottleneck_y - 5, 'black', 'solid')
attention_y = bottleneck_y - 5
draw_block(encoder_x, attention_y, 6, 4, colors['attention'],
           'Attention', 'Ch+Spatial', '(B,1024,14,14)')

# Embedding extraction (side branch)
emb_x = encoder_x + 12
emb_y = bottleneck_y
draw_arrow(encoder_x + 3, bottleneck_y, emb_x - 4, emb_y, colors['attention'], 'solid', 2)
draw_block(emb_x, emb_y, 7, 3, colors['attention'],
           'Embedding', 'Pool→FC', '(B,256)')

# Bridge from encoder to decoder
bridge_x = 42
draw_arrow(encoder_x + 3, attention_y, bridge_x, attention_y, 'black', 'solid', 2.5)

# Decoder stages (going upward)
decoder_stages = [
    {'ch': 512, 'size': 28},
    {'ch': 256, 'size': 56},
    {'ch': 128, 'size': 112},
    {'ch': 64, 'size': 224}
]

decoder_y = attention_y
for i, stage in enumerate(decoder_stages):
    # UpConv
    upconv_x = bridge_x + i * 7
    dim_str = f"(B,{stage['ch']},{stage['size']},{stage['size']})"

    if i == 0:
        draw_arrow(bridge_x, decoder_y, upconv_x - 3.5, decoder_y, 'black', 'solid')
    else:
        prev_x = bridge_x + (i-1) * 7
        draw_arrow(prev_x, decoder_y, upconv_x - 3.5, decoder_y, 'black', 'solid')

    draw_block(upconv_x, decoder_y, 6, 3.5, colors['upconv'],
               f'UpConv {i+1}', 'ConvT2D', dim_str)

    # Skip connection
    skip_idx = len(skip_positions) - 1 - i
    draw_curved_skip(skip_positions[skip_idx][0], skip_positions[skip_idx][1],
                    upconv_x - 2, decoder_y + 2, colors['skip'])

    # Arrow to concat
    draw_arrow(upconv_x, decoder_y + 1.75, upconv_x, decoder_y + 4, 'black', 'solid')
    decoder_y += 4

    # Concatenation
    concat_dim = f"(B,{stage['ch']*2},{stage['size']},{stage['size']})"
    draw_block(upconv_x, decoder_y, 6, 2.5, colors['concat'],
               'Concat', 'Skip+Up', concat_dim)

    # Arrow to conv
    draw_arrow(upconv_x, decoder_y + 1.25, upconv_x, decoder_y + 3, 'black', 'solid')
    decoder_y += 3

    # Conv block
    conv_dim = f"(B,{stage['ch']},{stage['size']},{stage['size']})"
    draw_block(upconv_x, decoder_y, 6, 4, colors['conv'],
               f'Conv Block {i+1}', '2×Conv3×3', conv_dim)

    # Move back down for next iteration
    decoder_y = attention_y

# Final output
final_x = bridge_x + len(decoder_stages) * 7
final_y = decoder_y + 11
last_conv_x = bridge_x + (len(decoder_stages) - 1) * 7

draw_arrow(last_conv_x, decoder_y + 2, last_conv_x, final_y - 2, 'black', 'solid')
draw_block(last_conv_x, final_y, 6, 4, colors['output'],
           'Output', 'Conv1×1', '(B,3,224,224)')

# Annotations
# Compression
ax.annotate('Compression\n16×', xy=(encoder_x - 5, (start_y + attention_y)/2),
            fontsize=11, fontweight='bold', color=colors['attention'],
            ha='center', va='center', rotation=90)

# Reconstruction
ax.annotate('Reconstruction', xy=(final_x + 8, (start_y + final_y)/2),
            fontsize=11, fontweight='bold', color=colors['upconv'],
            ha='center', va='center', rotation=90)

# Legend
legend_x = 85
legend_y = 75

ax.text(legend_x, legend_y + 2, 'Legend', fontsize=14, fontweight='bold',
        ha='left', va='top', color='#2C3E50')

legend_items = [
    (colors['input'], 'Input/Output'),
    (colors['conv'], 'Conv Block'),
    (colors['pool'], 'MaxPool'),
    (colors['bottleneck'], 'Bottleneck'),
    (colors['attention'], 'Attention'),
    (colors['upconv'], 'Upsampling'),
    (colors['concat'], 'Concat'),
    (colors['skip'], 'Skip Conn.')
]

for i, (color, label) in enumerate(legend_items):
    y = legend_y - i * 3
    box = mpatches.Rectangle((legend_x, y - 0.8), 1.5, 1.5,
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(legend_x + 2.5, y, label, fontsize=10, va='center', color='#2C3E50')

# Architecture details
details_y = legend_y - len(legend_items) * 3 - 4
ax.text(legend_x, details_y, 'Architecture Details', fontsize=12,
        fontweight='bold', ha='left', color='#2C3E50')

details = [
    'Input: 224×224 LAB + Mask (4ch)',
    'Encoder: 4 stages [64,128,256,512]',
    'Bottleneck: 1024ch @ 14×14',
    'Attention: Channel + Spatial',
    'Decoder: 4 stages + skip conn.',
    'Output: 224×224 LAB (3ch)',
    'Embedding: 256-dim vector',
    '',
    'Compression ratio: 16×',
    'Architecture: U-Net style'
]

for i, detail in enumerate(details):
    ax.text(legend_x, details_y - 1.8 - i * 1.8, detail,
            fontsize=9, ha='left', color='#34495E')

# Add a subtle grid for reference (optional, very light)
ax.grid(False)

plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('architecture_diagram.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✓ Created architecture_diagram.png (PNG, 150 DPI)")
print("✓ Created architecture_diagram.svg (SVG, vector format)")
print("\nDiagram features:")
print("  - 16:9 aspect ratio (presentation-ready)")
print("  - Horizontal flow: input → encoder → bottleneck → decoder → output")
print("  - Color-coded layer types")
print("  - Dimension labels at each transformation")
print("  - Skip connections shown with dashed curves")
print("  - Attention mechanism highlighted")
print("  - Embedding extraction branch")
