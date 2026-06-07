import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
import torchvision.models as models
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── SSIM ──────────────────────────────────────────────────────────────────────

def compute_ssim(img_path1, img_path2, size=384):
    """Compute SSIM between two images. Higher = more similar."""
    img1 = np.array(Image.open(img_path1).convert("RGB").resize((size, size)))
    img2 = np.array(Image.open(img_path2).convert("RGB").resize((size, size)))
    score = ssim(img1, img2, channel_axis=2, data_range=255)
    return round(score, 4)


# ── FID ───────────────────────────────────────────────────────────────────────

def get_inception_features(img_path, model, size=299):
    """Extract InceptionV3 features from a single image."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().cpu().numpy()


def compute_fid_single(result_path, style_path):
    """
    Approximate FID between result and style image using InceptionV3 features.
    Lower = result looks more like the style image (more stylised).
    Note: True FID requires many images; this is a single-image approximation
    used for relative comparison across ablation runs.
    """
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # remove final classification layer
    model = model.to(DEVICE).eval()

    f1 = get_inception_features(result_path, model)
    f2 = get_inception_features(style_path, model)

    # Euclidean distance in feature space as FID proxy
    dist = np.linalg.norm(f1 - f2)
    return round(float(dist), 2)


# ── Ablation Evaluation ───────────────────────────────────────────────────────

def evaluate_ablation(
    content_path="images/content/content.jpg",
    style_path="images/style/style.jpg",
    ablation_dir="results/ablation"
):
    style_weights = [1e6, 1e7, 1e8, 1e9, 1e10]

    print("Loading InceptionV3 for FID approximation...")
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE).eval()

    print("\n" + "=" * 65)
    print(f"{'Style Weight':<15} {'SSIM vs Content':<20} {'FID vs Style':<20}")
    print("=" * 65)

    ssim_scores = []
    fid_scores  = []
    labels      = []

    for sw in style_weights:
        result_path = os.path.join(ablation_dir, f"style_weight_{sw:.0e}.png")

        if not os.path.exists(result_path):
            print(f"  {sw:.0e} — file not found, skipping")
            continue

        # SSIM vs content (higher = more content preserved)
        ssim_score = compute_ssim(result_path, content_path)

        # FID proxy vs style (lower = more stylised)
        f_result = get_inception_features(result_path, model)
        f_style  = get_inception_features(style_path, model)
        fid_score = round(float(np.linalg.norm(f_result - f_style)), 2)

        ssim_scores.append(ssim_score)
        fid_scores.append(fid_score)
        labels.append(f"{sw:.0e}")

        print(f"  {sw:<15.0e} {ssim_score:<20} {fid_score:<20}")

    print("=" * 65)
    print(f"\nBest content preservation (highest SSIM): sw={labels[ssim_scores.index(max(ssim_scores))]}")
    print(f"Most stylised (lowest FID vs style):      sw={labels[fid_scores.index(min(fid_scores))]}")

    # Save evaluation chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(labels, ssim_scores, color='steelblue')
    ax1.set_title("SSIM vs Content Image\n(higher = more content preserved)")
    ax1.set_xlabel("Style Weight")
    ax1.set_ylabel("SSIM Score")
    ax1.set_ylim(0, 1)

    ax2.bar(labels, fid_scores, color='coral')
    ax2.set_title("FID Proxy vs Style Image\n(lower = more stylised)")
    ax2.set_xlabel("Style Weight")
    ax2.set_ylabel("Feature Distance")

    plt.suptitle("Ablation Study: SSIM + FID Evaluation", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/ablation/evaluation_charts.png", dpi=150)
    plt.close()
    print("\nEvaluation charts saved → results/ablation/evaluation_charts.png")

    return labels, ssim_scores, fid_scores


if __name__ == "__main__":
    evaluate_ablation(
        content_path="images/content/content.jpg",
        style_path="images/style/style.jpg",
        ablation_dir="results/ablation"
    )