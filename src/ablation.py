import torch
import torch.optim as optim
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_image, save_image, unnormalize, DEVICE
from model import VGG19Features, content_loss, style_loss

def run_single(content_img, style_img, model, content_weight, style_weight,
               num_steps=200):
    generated = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=0.01)

    with torch.no_grad():
        content_features = model(content_img)
        style_features   = model(style_img)

    final_c_loss = 0
    final_s_loss = 0

    for step in range(num_steps):
        optimizer.zero_grad()
        gen_features = model(generated)
        c_loss = content_loss(gen_features, content_features)
        s_loss = style_loss(gen_features, style_features)
        total  = content_weight * c_loss + style_weight * s_loss
        total.backward()
        optimizer.step()

        if step == num_steps - 1:
            final_c_loss = c_loss.item()
            final_s_loss = s_loss.item()

    return generated.detach(), final_c_loss, final_s_loss


def run_ablation(
    content_path="images/content/content.jpg",
    style_path="images/style/style.jpg",
    num_steps=200
):
    os.makedirs("results/ablation", exist_ok=True)

    # Ablation 1: vary style weight, fix content weight
    style_weights  = [1e6, 1e7, 1e8, 1e9, 1e10]
    content_weight = 1e4

    print("=" * 50)
    print("ABLATION: Style Weight Sweep")
    print("=" * 50)

    content_img = load_image(content_path)
    style_img   = load_image(style_path)
    model       = VGG19Features().to(DEVICE)

    results     = []
    c_losses    = []
    s_losses    = []

    for sw in style_weights:
        print(f"\nRunning style_weight={sw:.0e} ...")
        gen, cl, sl = run_single(content_img, style_img, model,
                                  content_weight, sw, num_steps)
        results.append(gen)
        c_losses.append(cl)
        s_losses.append(sl)
        save_image(gen, f"results/ablation/style_weight_{sw:.0e}.png")
        print(f"  Content Loss: {cl:.4f} | Style Loss: {sl:.6f}")

    # Save comparison grid
    fig, axes = plt.subplots(1, len(style_weights), figsize=(20, 4))
    fig.suptitle("Ablation: Style Weight Sweep (content_weight=1e4)", fontsize=13)

    import torchvision.transforms as transforms
    to_pil = transforms.ToPILImage()

    for ax, gen, sw in zip(axes, results, style_weights):
        img = unnormalize(gen.unsqueeze(0)).clamp(0, 1).cpu()
        ax.imshow(to_pil(img.squeeze(0)))
        ax.set_title(f"sw={sw:.0e}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/ablation/style_weight_comparison.png", dpi=150)
    plt.close()
    print("\nGrid saved → results/ablation/style_weight_comparison.png")

    # Save loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    labels = [f"{sw:.0e}" for sw in style_weights]

    ax1.bar(labels, c_losses, color='steelblue')
    ax1.set_title("Content Loss vs Style Weight")
    ax1.set_xlabel("Style Weight")
    ax1.set_ylabel("Content Loss")

    ax2.bar(labels, s_losses, color='coral')
    ax2.set_title("Style Loss vs Style Weight")
    ax2.set_xlabel("Style Weight")
    ax2.set_ylabel("Style Loss")

    plt.tight_layout()
    plt.savefig("results/ablation/loss_curves.png", dpi=150)
    plt.close()
    print("Loss curves saved → results/ablation/loss_curves.png")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Style Weight':<15} {'Content Loss':<20} {'Style Loss':<20}")
    print("=" * 60)
    for sw, cl, sl in zip(style_weights, c_losses, s_losses):
        print(f"{sw:<15.0e} {cl:<20.4f} {sl:<20.6f}")
    print("=" * 60)
    print("\nAblation complete! Check results/ablation/")


if __name__ == "__main__":
    run_ablation(
        content_path="images/content/content.jpg",
        style_path="images/style/style.jpg",
        num_steps=200
    )