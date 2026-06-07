# 🎨 Neural Style Transfer + Moodboard Generator

A research implementation of Gatys et al. (2015) "A Neural Algorithm of Artistic Style" in PyTorch, extended with multi-image moodboard blending, CLIP text conditioning, K-means colour palette extraction, and ablation studies with SSIM + FID evaluation.

Built as a portfolio project targeting generative AI and Content Intelligence research — directly mirroring Adobe Firefly's Match Style feature.

---

## 🎯 Project Highlights

- Reproduced Gatys et al. (2015) Neural Style Transfer using VGG19 Gram matrix style representations
- Extended to multi-image moodboard blending aggregating style embeddings from 2–5 references
- Integrated CLIP ViT-B/32 text conditioning to steer optimisation with prompts like "warm, vintage, cinematic"
- Added K-means colour palette extraction with SSIM and FID evaluation
- Conducted ablation study on content/style loss weightings across 5 style weights
- Built full Gradio UI with 6 tabs — Style Transfer, Moodboard, CLIP, Palette, Ablation, Evaluation

---

## 🖼️ Results

### Style Transfer — NYC Skyline + Van Gogh Starry Night
| Content | Style | Result |
|---|---|---|
| NYC Skyline | Van Gogh Starry Night | Van Gogh brush strokes on cityscape |

### Ablation Study Results
Style weight sweep with fixed `content_weight=1e4`, 200 optimisation steps:

| Style Weight | SSIM vs Content | FID vs Style | Interpretation |
|---|---|---|---|
| 1e6  | 0.9500 | 25.08 | Almost no style — content fully preserved |
| 1e7  | 0.9193 | 24.59 | Slight texture, content dominant |
| 1e8  | 0.7678 | 23.85 | **Sweet spot** — style and content balanced |
| 1e9  | 0.7524 | 22.97 | Heavy style, content fading |
| 1e10 | 0.7519 | 22.64 | Maximum style, content barely preserved |

**Key finding:** SSIM drops sharply from 0.92 → 0.77 between style weights 1e7 and 1e8, identifying **1e8 as the optimal style weight** — the point where style begins to dominate content structure.

## 🏗️ Project Structure

    neural-style-transfer-moodboard/
    ├── app.py                         # Gradio UI (6 tabs)
    ├── main.py                        # CLI entry point
    ├── src/
    │   ├── utils.py                   # Image loading, saving, display
    │   ├── model.py                   # VGG19 features, Gram matrix, losses
    │   ├── style_transfer.py          # Core style transfer loop
    │   ├── moodboard.py               # Multi-image style blending (CPU)
    │   ├── clip_conditioning.py       # CLIP text-conditioned style transfer
    │   ├── ablation.py                # Ablation study over style weights
    │   ├── palette.py                 # K-means colour palette extraction
    │   └── evaluation.py             # SSIM + FID evaluation metrics
    ├── images/
    │   ├── content/                   # Content images
    │   └── style/                     # Style images
    └── results/
        ├── output.png                 # Style transfer result
        ├── moodboard_output.png       # Moodboard result
        ├── clip_output.png            # CLIP-conditioned result
        ├── comparison.png             # Side by side comparison
        ├── palette_comparison.png     # Colour palette comparison
        └── ablation/
            ├── style_weight_comparison.png
            ├── loss_curves.png
            └── evaluation_charts.png

## ⚙️ Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA (tested on GTX 1650 4GB, CUDA 12.5)
- 4GB+ VRAM recommended

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pillow matplotlib numpy scikit-learn scikit-image scipy gradio
pip install git+https://github.com/openai/CLIP.git
```

---

## 🚀 Usage

### Gradio UI
```bash
python app.py
```
Opens at `http://127.0.0.1:7860` with 6 tabs:
- **Style Transfer** — VGG19 Gram matrix optimisation
- **Moodboard Blending** — multi-image style aggregation
- **CLIP Conditioning** — text-steered style transfer
- **Colour Palette** — K-means palette extraction
- **Ablation Study** — automated style weight sweep
- **Evaluation** — SSIM + FID metrics

### CLI
```bash
# Basic style transfer
python main.py --mode style --content images/content/content.jpg --style images/style/style.jpg

# Moodboard blending
python main.py --mode moodboard --styles images/style/style.jpg images/style/style2.jpg --style_weights 0.6 0.4

# CLIP text conditioning
python main.py --mode clip --prompt "warm, vintage, cinematic"

# Ablation study
python main.py --mode ablation

# Colour palette
python main.py --mode palette --output results/output.png
```

---

## 🔬 Technical Details

### Neural Style Transfer
- Feature extractor: VGG19 pretrained on ImageNet
- Content layers: `conv4_2`
- Style layers: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- Style representation: Gram matrices (channel correlation maps)
- Optimiser: Adam (lr=0.01)
- Image size: 384×384 (optimised for 4GB VRAM)

### Moodboard Blending
- Aggregates Gram matrices from 2–5 style references with custom weights
- Blended Gram = Σ(wᵢ × Gram(styleᵢ)) normalised to sum to 1
- Runs on CPU to avoid VRAM fragmentation

### CLIP Text Conditioning
- Model: CLIP ViT-B/32
- Text features extracted once before optimisation, then CLIP deleted from memory
- CLIP direction loss = negative cosine similarity between generated image and text direction
- Runs on CPU; VGG19 optimisation on GPU

### Evaluation Metrics
- **SSIM** (Structural Similarity Index) — measures content preservation vs original photo
- **FID proxy** — InceptionV3 feature distance between result and style image (single-image approximation)

---

## 🖥️ Hardware

Developed and tested on:
- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- CUDA: 12.5
- OS: Windows 11
- Python: 3.13

VRAM optimisations:
- Image size reduced to 384×384
- Moodboard runs on CPU
- CLIP loaded and deleted before VGG19 optimisation
- `torch.cuda.empty_cache()` between runs

---

## 📚 References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). arXiv:1508.06576
- Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP). ICML 2021
- Simonyan & Zisserman (2014). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (VGG19)

---

## 👩‍💻 Author

**Avantika Gurav**
third year B.Tech (Information Technology) — Cummins College of Engineering for Women, Pune
[LinkedIn](https://linkedin.com/in/avantika-gurav) · [GitHub](https://github.com/Avantika580)
[Medium Blog Post](https://medium.com/@avantika9542/building-neural-style-transfer-from-scratch-what-i-learned-reproducing-a-2015-paper-on-a-4gb-gpu-6618b0c2125a)
