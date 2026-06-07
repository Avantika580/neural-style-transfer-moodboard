import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer + Moodboard Generator"
    )
    parser.add_argument("--mode", type=str, required=True,
        choices=["style", "moodboard", "clip", "ablation", "palette"],
        help="Which mode to run")

    # Shared args
    parser.add_argument("--content",  type=str, default="images/content/content.jpg")
    parser.add_argument("--style",    type=str, default="images/style/style.jpg")
    parser.add_argument("--output",   type=str, default="results/output.png")
    parser.add_argument("--steps",    type=int, default=300)
    parser.add_argument("--content_weight", type=float, default=1e4)
    parser.add_argument("--style_weight",   type=float, default=1e10)

    # CLIP args
    parser.add_argument("--prompt", type=str, default="warm, vintage, cinematic")
    parser.add_argument("--clip_weight", type=float, default=30.0)

    # Moodboard args
    parser.add_argument("--styles", type=str, nargs="+",
        default=["images/style/style.jpg", "images/style/style2.jpg"],
        help="List of style images for moodboard")
    parser.add_argument("--style_weights", type=float, nargs="+",
        default=None,
        help="Weights for each style image in moodboard")

    args = parser.parse_args()

    if args.mode == "style":
        print("Running Neural Style Transfer...")
        from src.style_transfer import run_style_transfer
        run_style_transfer(
            content_path=args.content,
            style_path=args.style,
            output_path=args.output,
            num_steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight
        )

    elif args.mode == "moodboard":
        print("Running Moodboard Blending...")
        from src.moodboard import run_moodboard
        run_moodboard(
            content_path=args.content,
            style_paths=args.styles,
            style_weights=args.style_weights,
            output_path=args.output,
            num_steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight
        )

    elif args.mode == "clip":
        print("Running CLIP-conditioned Style Transfer...")
        from src.clip_conditioning import run_clip_style_transfer
        run_clip_style_transfer(
            content_path=args.content,
            style_path=args.style,
            text_prompt=args.prompt,
            output_path=args.output,
            num_steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            clip_weight=args.clip_weight
        )

    elif args.mode == "ablation":
        print("Running Ablation Study...")
        from src.ablation import run_ablation
        run_ablation(
            content_path=args.content,
            style_path=args.style,
            num_steps=args.steps
        )

    elif args.mode == "palette":
        print("Extracting Colour Palettes...")
        from src.palette import compare_palettes
        compare_palettes(
            content_path=args.content,
            style_path=args.style,
            result_path=args.output
        )


if __name__ == "__main__":
    main()