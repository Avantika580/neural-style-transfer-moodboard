import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 384

def load_image(path, size=IMAGE_SIZE):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE)

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(DEVICE)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(DEVICE)
    return tensor.squeeze(0) * std + mean

def save_image(tensor, path):
    image = unnormalize(tensor).clamp(0, 1)
    image = transforms.ToPILImage()(image.cpu())
    image.save(path)
    print(f"Saved → {path}")

def show_images(content_path, style_path, result_tensor):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Content", "Style", "Result"]
    images = [
        Image.open(content_path).resize((512, 512)),
        Image.open(style_path).resize((512, 512)),
        transforms.ToPILImage()(unnormalize(result_tensor).clamp(0,1).cpu())
    ]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150)
    plt.show()
    print("Comparison saved → results/comparison.png")