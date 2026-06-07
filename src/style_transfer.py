import torch
import torch.optim as optim
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_image, save_image, show_images, DEVICE
from model import VGG19Features, content_loss, style_loss

def run_style_transfer(
    content_path,
    style_path,
    output_path="results/output.png",
    num_steps=500,
    content_weight=1e4,
    style_weight=1e10
):
    print(f"Using device: {DEVICE}")
    print("Loading images...")

    content_img = load_image(content_path)
    style_img   = load_image(style_path)

    # Start from content image
    generated = content_img.clone().requires_grad_(True)

    model = VGG19Features().to(DEVICE)

    optimizer = optim.Adam([generated], lr=0.01)

    # Extract fixed features
    content_features = model(content_img)
    style_features   = model(style_img)

    print(f"Starting optimization for {num_steps} steps...")

    for step in range(num_steps):
        optimizer.zero_grad()

        gen_features = model(generated)

        c_loss = content_loss(gen_features, content_features)
        s_loss = style_loss(gen_features, style_features)
        total  = content_weight * c_loss + style_weight * s_loss

        total.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:3d} | "
                  f"Content Loss: {c_loss.item():.4f} | "
                  f"Style Loss: {s_loss.item():.6f} | "
                  f"Total: {total.item():.2f}")

    print("\nOptimization complete!")
    save_image(generated, output_path)
    show_images(content_path, style_path, generated)


if __name__ == "__main__":
    run_style_transfer(
        content_path="images/content/content.jpg",
        style_path="images/style/style.jpg",
        output_path="results/output.png",
        num_steps=500,
        content_weight=1e4,
        style_weight=1e10
    )