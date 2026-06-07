import gradio as gr
import torch
import gc
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def run_style_transfer_ui(content_img, style_img, style_weight, content_weight, steps):
    if content_img is None or style_img is None:
        return None, "Please upload both content and style images."
    try:
        clear_vram()
        content_img.save("images/content/content_ui.jpg")
        style_img.save("images/style/style_ui.jpg")
        from src.style_transfer import run_style_transfer
        run_style_transfer(
            content_path="images/content/content_ui.jpg",
            style_path="images/style/style_ui.jpg",
            output_path="results/ui_output.png",
            num_steps=int(steps),
            content_weight=float(content_weight),
            style_weight=float(style_weight)
        )
        clear_vram()
        from PIL import Image
        return Image.open("results/ui_output.png"), "✅ Style transfer complete!"
    except Exception as e:
        clear_vram()
        return None, f"❌ Error: {str(e)}"

def run_moodboard_ui(content_img, style_img1, style_img2, style_img3,
                     weight1, weight2, weight3, steps, style_weight):
    if content_img is None or style_img1 is None or style_img2 is None:
        return None, "Please upload content image and at least 2 style images."
    try:
        clear_vram()
        content_img.save("images/content/content_ui.jpg")
        style_img1.save("images/style/style_ui1.jpg")
        style_img2.save("images/style/style_ui2.jpg")
        style_paths   = ["images/style/style_ui1.jpg", "images/style/style_ui2.jpg"]
        style_weights = [weight1, weight2]
        if style_img3 is not None:
            style_img3.save("images/style/style_ui3.jpg")
            style_paths.append("images/style/style_ui3.jpg")
            style_weights.append(weight3)
        from src.moodboard import run_moodboard
        run_moodboard(
            content_path="images/content/content_ui.jpg",
            style_paths=style_paths,
            style_weights=style_weights,
            output_path="results/ui_moodboard.png",
            num_steps=int(steps),
            content_weight=1e4,
            style_weight=float(style_weight)
        )
        clear_vram()
        from PIL import Image
        return Image.open("results/ui_moodboard.png"), "✅ Moodboard blending complete!"
    except Exception as e:
        clear_vram()
        return None, f"❌ Error: {str(e)}"

def run_clip_ui(content_img, style_img, prompt, steps, style_weight, clip_weight):
    if content_img is None or style_img is None:
        return None, "Please upload both images."
    if not prompt.strip():
        return None, "Please enter a text prompt."
    try:
        clear_vram()
        content_img.save("images/content/content_ui.jpg")
        style_img.save("images/style/style_ui.jpg")
        from src.clip_conditioning import run_clip_style_transfer
        run_clip_style_transfer(
            content_path="images/content/content_ui.jpg",
            style_path="images/style/style_ui.jpg",
            text_prompt=prompt,
            output_path="results/ui_clip.png",
            num_steps=int(steps),
            content_weight=1e4,
            style_weight=float(style_weight),
            clip_weight=float(clip_weight)
        )
        clear_vram()
        from PIL import Image
        return Image.open("results/ui_clip.png"), f"✅ CLIP conditioning complete! Prompt: '{prompt}'"
    except Exception as e:
        clear_vram()
        return None, f"❌ Error: {str(e)}"

def run_palette_ui(content_img, style_img, result_img):
    if content_img is None or style_img is None or result_img is None:
        return None, "Please upload all three images."
    try:
        content_img.save("images/content/content_ui.jpg")
        style_img.save("images/style/style_ui.jpg")
        result_img.save("results/palette_input.png")
        from src.palette import compare_palettes
        compare_palettes(
            content_path="images/content/content_ui.jpg",
            style_path="images/style/style_ui.jpg",
            result_path="results/palette_input.png"
        )
        from PIL import Image
        return Image.open("results/palette_comparison.png"), "✅ Palette extraction complete!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def run_ablation_ui(content_img, style_img, steps):
    if content_img is None or style_img is None:
        return None, None, "Please upload both images."
    try:
        clear_vram()
        content_img.save("images/content/content_ui.jpg")
        style_img.save("images/style/style_ui.jpg")
        from src.ablation import run_ablation
        run_ablation(
            content_path="images/content/content_ui.jpg",
            style_path="images/style/style_ui.jpg",
            num_steps=int(steps)
        )
        clear_vram()
        from PIL import Image
        grid   = Image.open("results/ablation/style_weight_comparison.png")
        curves = Image.open("results/ablation/loss_curves.png")
        return grid, curves, "✅ Ablation study complete!"
    except Exception as e:
        clear_vram()
        return None, None, f"❌ Error: {str(e)}"

def run_evaluation_ui(content_img, style_img):
    if content_img is None or style_img is None:
        return None, "Please upload both images."
    try:
        content_img.save("images/content/content_ui.jpg")
        style_img.save("images/style/style_ui.jpg")
        from src.evaluation import evaluate_ablation
        labels, ssim_scores, fid_scores = evaluate_ablation(
            content_path="images/content/content_ui.jpg",
            style_path="images/style/style_ui.jpg",
            ablation_dir="results/ablation"
        )
        table = "Style Weight | SSIM vs Content | FID vs Style\n"
        table += "-" * 45 + "\n"
        for l, s, f in zip(labels, ssim_scores, fid_scores):
            table += f"{l:<15} | {s:<15} | {f}\n"
        table += "-" * 45 + "\n"
        table += f"Best content preservation: sw={labels[ssim_scores.index(max(ssim_scores))]}\n"
        table += f"Most stylised:             sw={labels[fid_scores.index(min(fid_scores))]}\n"
        table += f"Optimal balance:           sw=1e+08 (SSIM drops sharply here)"
        from PIL import Image
        chart = Image.open("results/ablation/evaluation_charts.png")
        return chart, table
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


css = """
/* ── Global ── */
body, .gradio-container { font-family: 'Inter', sans-serif !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem; color: white; }
.hero p  { font-size: 1rem; opacity: 0.8; margin: 0 0 1rem; color: white; }
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: white;
}
.hero-note {
    background: rgba(255,193,7,0.15);
    border: 1px solid rgba(255,193,7,0.3);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.85rem;
    color: #ffc107;
    margin-top: 1rem;
}

/* ── Tabs ── */
.tab-nav button {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
}
.tab-nav button.selected {
    background: #0f3460 !important;
    color: white !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f3460;
    border-left: 4px solid #e94560;
    padding-left: 12px;
    margin-bottom: 1rem;
}

/* ── Info boxes ── */
.info-box {
    background: #f0f7ff;
    border: 1px solid #b3d4f5;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
.info-box table { width: 100%; border-collapse: collapse; }
.info-box th { text-align: left; padding: 6px 8px; background: #dbeeff; border-radius: 4px; }
.info-box td { padding: 6px 8px; border-bottom: 1px solid #dbeeff; }

/* ── Run buttons ── */
.run-btn {
    background: linear-gradient(135deg, #0f3460, #e94560) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    border: none !important;
}
.run-btn:hover { opacity: 0.9 !important; }

/* ── Status box ── */
.status-box textarea {
    background: #f8f9fa !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    font-family: monospace !important;
    font-size: 0.85rem !important;
}
"""

with gr.Blocks(title="Neural Style Transfer + Moodboard Generator", css=css) as demo:

    gr.HTML("""
    <div class="hero">
        <h1>🎨 Neural Style Transfer + Moodboard Generator</h1>
        <p>Reproduction of Gatys et al. (2015) extended with multi-image moodboard blending,
           CLIP text conditioning, K-means palette extraction, and ablation studies.
           Built to mirror Adobe Firefly's Match Style feature.</p>
        <div class="hero-badges">
            <span class="badge">VGG19 Gram Matrix</span>
            <span class="badge">CLIP ViT-B/32</span>
            <span class="badge">Multi-image Moodboard</span>
            <span class="badge">K-means Palette</span>
            <span class="badge">SSIM + FID Evaluation</span>
            <span class="badge">GTX 1650 · CUDA 12.5</span>
        </div>
        <div class="hero-note">
            ⚠️ Run one feature at a time. Restart the app between heavy runs to free GPU memory.
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes="tab-nav"):

        # ── Tab 1: Style Transfer ──────────────────────────────────────────
        with gr.Tab("🖼️ Style Transfer"):
            gr.HTML('<div class="info-box">Upload a <b>content photo</b> and a <b>style painting</b>. VGG19 extracts Gram matrix style representations and optimises the output image to match both. Based on Gatys et al. (2015).</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Inputs</p>')
                    st_content = gr.Image(type="pil", label="Content Image")
                    st_style   = gr.Image(type="pil", label="Style Image")
                    gr.HTML('<p class="section-header">Parameters</p>')
                    st_sw    = gr.Slider(1e6, 1e10, value=1e10, step=1e6, label="Style Weight — higher = more painting style")
                    st_cw    = gr.Slider(1e3, 1e5,  value=1e4,  step=1e3, label="Content Weight — higher = more photo preserved")
                    st_steps = gr.Slider(100, 500,  value=300,  step=50,  label="Optimisation Steps — more = better quality")
                    st_btn   = gr.Button("▶ Run Style Transfer", elem_classes="run-btn")
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Result</p>')
                    st_output = gr.Image(label="Stylised Output")
                    st_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
            st_btn.click(run_style_transfer_ui,
                         inputs=[st_content, st_style, st_sw, st_cw, st_steps],
                         outputs=[st_output, st_status])

        # ── Tab 2: Moodboard ───────────────────────────────────────────────
        with gr.Tab("🎭 Moodboard Blending"):
            gr.HTML('<div class="info-box">Blend style from <b>2–3 reference images</b> using weighted Gram matrix aggregation. Runs on CPU to avoid VRAM conflicts. Use <b>100 steps</b> to avoid timeout.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Inputs</p>')
                    mb_content = gr.Image(type="pil", label="Content Image")
                    mb_style1  = gr.Image(type="pil", label="Style Image 1")
                    mb_style2  = gr.Image(type="pil", label="Style Image 2")
                    mb_style3  = gr.Image(type="pil", label="Style Image 3 (optional)")
                    gr.HTML('<p class="section-header">Blend Weights</p>')
                    mb_w1    = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Weight — Style 1")
                    mb_w2    = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="Weight — Style 2")
                    mb_w3    = gr.Slider(0.1, 1.0, value=0.2, step=0.1, label="Weight — Style 3")
                    mb_steps = gr.Slider(50, 200, value=100, step=50, label="Steps (100 recommended)")
                    mb_sw    = gr.Slider(1e6, 1e10, value=1e10, step=1e6, label="Style Weight")
                    mb_btn   = gr.Button("▶ Run Moodboard Blending", elem_classes="run-btn")
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Result</p>')
                    mb_output = gr.Image(label="Blended Output")
                    mb_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
            mb_btn.click(run_moodboard_ui,
                         inputs=[mb_content, mb_style1, mb_style2, mb_style3,
                                 mb_w1, mb_w2, mb_w3, mb_steps, mb_sw],
                         outputs=[mb_output, mb_status])

        # ── Tab 3: CLIP Conditioning ───────────────────────────────────────
        with gr.Tab("✨ CLIP Conditioning"):
            gr.HTML('<div class="info-box">Steer the style transfer using a <b>text prompt</b> via CLIP ViT-B/32. Text features are extracted once then deleted to free memory before VGG19 optimisation begins.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Inputs</p>')
                    cl_content = gr.Image(type="pil", label="Content Image")
                    cl_style   = gr.Image(type="pil", label="Style Image")
                    cl_prompt  = gr.Textbox(value="warm, vintage, cinematic", label="Text Prompt — describe the mood you want")
                    gr.HTML('<p class="section-header">Parameters</p>')
                    cl_steps = gr.Slider(100, 500, value=300, step=50,  label="Optimisation Steps")
                    cl_sw    = gr.Slider(1e6, 1e10, value=1e10, step=1e6, label="Style Weight")
                    cl_cw    = gr.Slider(10.0, 100.0, value=30.0, step=10.0, label="CLIP Weight — higher = stronger text influence")
                    cl_btn   = gr.Button("▶ Run CLIP Style Transfer", elem_classes="run-btn")
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Result</p>')
                    cl_output = gr.Image(label="CLIP-conditioned Output")
                    cl_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
            cl_btn.click(run_clip_ui,
                         inputs=[cl_content, cl_style, cl_prompt, cl_steps, cl_sw, cl_cw],
                         outputs=[cl_output, cl_status])

        # ── Tab 4: Colour Palette ──────────────────────────────────────────
        with gr.Tab("🎨 Colour Palette"):
            gr.HTML('<div class="info-box">Extract dominant colours using <b>K-means clustering</b> (k=6) and compare palettes across content, style, and result images side by side.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Inputs</p>')
                    pa_content = gr.Image(type="pil", label="Content Image")
                    pa_style   = gr.Image(type="pil", label="Style Image")
                    pa_result  = gr.Image(type="pil", label="Result Image (from Style Transfer)")
                    pa_btn     = gr.Button("▶ Extract Colour Palettes", elem_classes="run-btn")
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Palette Comparison</p>')
                    pa_output = gr.Image(label="Content vs Style vs Result")
                    pa_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
            pa_btn.click(run_palette_ui,
                         inputs=[pa_content, pa_style, pa_result],
                         outputs=[pa_output, pa_status])

        # ── Tab 5: Ablation Study ──────────────────────────────────────────
        with gr.Tab("📊 Ablation Study"):
            gr.HTML('<div class="info-box">Automatically runs style transfer across <b>5 style weights</b> (1e6 → 1e10) and generates a comparison grid + loss curves. Run this before Evaluation.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Inputs</p>')
                    ab_content = gr.Image(type="pil", label="Content Image")
                    ab_style   = gr.Image(type="pil", label="Style Image")
                    ab_steps   = gr.Slider(50, 200, value=100, step=50, label="Steps per run (5 runs total)")
                    ab_btn     = gr.Button("▶ Run Ablation Study", elem_classes="run-btn")
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Results</p>')
                    ab_grid   = gr.Image(label="Style Weight Comparison Grid")
                    ab_curves = gr.Image(label="Loss Curves")
                    ab_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
            ab_btn.click(run_ablation_ui,
                         inputs=[ab_content, ab_style, ab_steps],
                         outputs=[ab_grid, ab_curves, ab_status])

        # ── Tab 6: Evaluation ─────────────────────────────────────────────
        with gr.Tab("📈 Evaluation"):
            gr.HTML("""
            <div class="info-box">
                <b>Run Ablation Study first</b>, then evaluate results here using SSIM + FID metrics.
                <table style="margin-top:10px;">
                    <tr><th>Metric</th><th>What it measures</th><th>Good value</th></tr>
                    <tr><td><b>SSIM vs Content</b></td><td>How much of the original photo is preserved</td><td>Higher = more photo visible</td></tr>
                    <tr><td><b>FID vs Style</b></td><td>How close the result looks to the style painting</td><td>Lower = more stylised</td></tr>
                </table>
                <p style="margin-top:10px; margin-bottom:0;">
                📌 <b>Sweet spot</b> = where SSIM drops sharply between two weights — that's your optimal style weight.
                </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Inputs</p>')
                    ev_content = gr.Image(type="pil", label="Content Image")
                    ev_style   = gr.Image(type="pil", label="Style Image")
                    ev_btn     = gr.Button("▶ Run Evaluation", elem_classes="run-btn")
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-header">Results</p>')
                    ev_chart  = gr.Image(label="SSIM + FID Charts")
                    ev_status = gr.Textbox(label="Results Table", interactive=False,
                                          lines=10, elem_classes="status-box")
            ev_btn.click(run_evaluation_ui,
                         inputs=[ev_content, ev_style],
                         outputs=[ev_chart, ev_status])

if __name__ == "__main__":
    demo.launch(share=False, max_threads=1)