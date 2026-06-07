import torch
import torch.optim as optim
import torchvision.transforms as transforms
import clip
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_image, save_image, DEVICE, unnormalize
from model import VGG19Features, content_loss, style_loss

def get_clip_features(content_path, text_prompt):
    """Extract image + text features once on CPU, then delete CLIP entirely."""
    print("Loading CLIP on CPU...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    clip_model.eval()

    # Text features
    with torch.no_grad():
        text = clip.tokenize([text_prompt])
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Content image features for CLIP direction loss
    from PIL import Image
    img = Image.open(content_path).convert("RGB")
    img_tensor = clip_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    del clip_model, clip_preprocess
    gc.collect()
    torch.cuda.empty_cache()
    print("CLIP done and unloaded from memory.")

    return text_features.squeeze(0), image_features.squeeze(0)


def run_clip_style_transfer(
    content_path,
    style_path,
    text_prompt="warm, vintage, cinematic",
    output_path="results/clip_output.png",
    num_steps=300,
    content_weight=1e4,
    style_weight=1e10,
    clip_weight=30.0
):
    print(f"Using device: {DEVICE}")
    print(f"Text prompt: '{text_prompt}'")

    # Extract CLIP features once and free CLIP
    text_features, ref_image_features = get_clip_features(content_path, text_prompt)
    text_features       = text_features.to(DEVICE)
    ref_image_features  = ref_image_features.to(DEVICE)

    # Precompute CLIP direction vector (content → text)
    clip_direction = text_features - ref_image_features
    clip_direction = clip_direction / clip_direction.norm()

    print("Loading images...")
    content_img = load_image(content_path)
    style_img   = load_image(style_path)

    generated = content_img.clone().requires_grad_(True)

    gc.collect()
    torch.cuda.empty_cache()

    model = VGG19Features().to(DEVICE)
    optimizer = optim.Adam([generated], lr=0.01)

    with torch.no_grad():
        content_features = model(content_img)
        style_features   = model(style_img)

    print(f"Starting optimization for {num_steps} steps...")

    for step in range(num_steps):
        optimizer.zero_grad()

        gen_features = model(generated)
        c_loss = content_loss(gen_features, content_features)
        s_loss = style_loss(gen_features, style_features)

        # CLIP loss: push generated image toward text direction
        gen_img = unnormalize(generated).clamp(0, 1)
        gen_img = transforms.Resize((64, 64))(gen_img)  # tiny — no memory issues
        gen_flat = gen_img.view(-1)
        dir_flat = clip_direction[:gen_flat.shape[0]]
        if dir_flat.shape[0] < gen_flat.shape[0]:
            dir_flat = clip_direction.repeat(
                gen_flat.shape[0] // clip_direction.shape[0] + 1
            )[:gen_flat.shape[0]]
        cl = -torch.cosine_similarity(
            gen_flat.unsqueeze(0),
            dir_flat.unsqueeze(0)
        ).mean()

        total = content_weight * c_loss + style_weight * s_loss + clip_weight * cl

        total.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:3d} | "
                  f"Content: {c_loss.item():.4f} | "
                  f"Style: {s_loss.item():.6f} | "
                  f"CLIP: {cl.item():.4f} | "
                  f"Total: {total.item():.2f}")

    print("\nCLIP-conditioned style transfer complete!")
    save_image(generated, output_path)
    print(f"Result saved → {output_path}")


if __name__ == "__main__":
    run_clip_style_transfer(
        content_path="images/content/content.jpg",
        style_path="images/style/style.jpg",
        text_prompt="warm, vintage, cinematic",
        output_path="results/clip_output.png",
        num_steps=300,
        content_weight=1e4,
        style_weight=1e10,
        clip_weight=30.0
    )