import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def extract_palette(image_path, n_colors=6):
    """Extract dominant colours from an image using K-means."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))  # smaller = faster
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colours = kmeans.cluster_centers_.astype(int)
    counts  = np.bincount(kmeans.labels_)
    # Sort by frequency
    sorted_idx = np.argsort(-counts)
    colours = colours[sorted_idx]
    counts  = counts[sorted_idx]
    percentages = counts / counts.sum() * 100

    return colours, percentages


def save_palette(colours, percentages, output_path="results/palette.png", title="Colour Palette"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)

    x = 0
    for colour, pct in zip(colours, percentages):
        hex_col = "#{:02x}{:02x}{:02x}".format(*colour)
        rect = patches.Rectangle((x, 0), pct, 1, color=hex_col)
        ax.add_patch(rect)
        ax.text(x + pct/2, 0.5, f"{pct:.1f}%",
                ha='center', va='center',
                fontsize=8, color='white' if sum(colour) < 400 else 'black',
                fontweight='bold')
        x += pct

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Palette saved → {output_path}")
    return fig


def compare_palettes(content_path, style_path, result_path):
    """Compare palettes of content, style and result side by side."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 5))
    labels = ["Content", "Style", "Result"]
    paths  = [content_path, style_path, result_path]

    for ax, path, label in zip(axes, paths, labels):
        colours, percentages = extract_palette(path, n_colors=6)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_ylabel(label, fontsize=11, rotation=0, labelpad=50, va='center')

        x = 0
        for colour, pct in zip(colours, percentages):
            hex_col = "#{:02x}{:02x}{:02x}".format(*colour)
            rect = patches.Rectangle((x, 0), pct, 1, color=hex_col)
            ax.add_patch(rect)
            x += pct

    plt.suptitle("Colour Palette Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("results/palette_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Palette comparison saved → results/palette_comparison.png")


if __name__ == "__main__":
    print("Extracting palettes...")

    compare_palettes(
        content_path="images/content/content.jpg",
        style_path="images/style/style.jpg",
        result_path="results/output.png"
    )

    # Also extract palette from CLIP result
    colours, pcts = extract_palette("results/clip_output.png", n_colors=6)
    save_palette(colours, pcts,
                 output_path="results/clip_palette.png",
                 title="CLIP Output Palette — warm, vintage, cinematic")

    print("\nDone! Check results/ folder.")