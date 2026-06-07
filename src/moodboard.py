import torch
import torch.optim as optim
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_image, save_image
from model import VGG19Features, content_loss, gram_matrix

DEVICE = "cpu"  # CPU to avoid VRAM fragmentation in UI

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

def load_image_cpu(path):
    from PIL import Image
    import torchvision.transforms as transforms
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def save_image_cpu(tensor, path):
    import torchvision.transforms as transforms
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = (tensor.squeeze(0) * std + mean).clamp(0, 1)
    image = transforms.ToPILImage()(image.cpu())
    image.save(path)
    print(f"Saved → {path}")

def blended_style_loss(gen_features, style_features_list, weights):
    loss = 0
    for layer in STYLE_LAYERS:
        G = gram_matrix(gen_features[layer])
        blended_gram = sum(
            w * gram_matrix(sf[layer])
            for w, sf in zip(weights, style_features_list)
        )
        loss += torch.mean((G - blended_gram) ** 2)
    return loss


def run_moodboard(
    content_path,
    style_paths,
    style_weights=None,
    output_path="results/moodboard_output.png",
    num_steps=150,
    content_weight=1e4,
    style_weight=1e10
):
    print(f"Using device: {DEVICE}")
    print(f"Blending {len(style_paths)} style images...")

    if style_weights is None:
        style_weights = [1.0 / len(style_paths)] * len(style_paths)

    total_w = sum(style_weights)
    style_weights = [w / total_w for w in style_weights]

    content_img = load_image_cpu(content_path)
    style_imgs  = [load_image_cpu(p) for p in style_paths]

    generated = content_img.clone().requires_grad_(True)

    # Run VGG19 on CPU
    model = VGG19Features().to(DEVICE)
    optimizer = optim.Adam([generated], lr=0.01)

    with torch.no_grad():
        content_features    = model(content_img)
        style_features_list = [model(s) for s in style_imgs]

    print(f"Starting optimization for {num_steps} steps...")

    for step in range(num_steps):
        optimizer.zero_grad()

        gen_features = model(generated)
        c_loss = content_loss(gen_features, content_features)
        s_loss = blended_style_loss(gen_features, style_features_list, style_weights)
        total  = content_weight * c_loss + style_weight * s_loss

        total.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:3d} | "
                  f"Content Loss: {c_loss.item():.4f} | "
                  f"Style Loss: {s_loss.item():.6f} | "
                  f"Total: {total.item():.2f}")

    print("\nMoodboard optimization complete!")
    save_image_cpu(generated, output_path)
    print(f"Result saved → {output_path}")


if __name__ == "__main__":
    run_moodboard(
        content_path="images/content/content.jpg",
        style_paths=[
            "images/style/style.jpg",
            "images/style/style2.jpg",
        ],
        style_weights=[0.6, 0.4],
        output_path="results/moodboard_output.png",
        num_steps=150,
        content_weight=1e4,
        style_weight=1e10
    )