"""
Artist Style Erasure Evaluation via CLIP Similarity.

Compares CLIP text-image similarity between original and concept-erased
models to verify that the erased model produces images less aligned with
the target artist's style prompts.

Usage:
    python artist_erasure.py \
        --ckpt_name path/to/pruned_unet.pt \
        --target "Van Gogh" \
        --output_dir results/artist_eval
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------------------------
# Model loading (self-contained)
# ---------------------------------------------------------------------------
def load_erased_model(model_id: str, ckpt_name: str | None, gpu: int):
    """Load a Stable Diffusion pipeline, optionally replacing the UNet."""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if ckpt_name is not None:
        state_dict = torch.load(ckpt_name, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)
    return pipe.to(gpu)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--ckpt_name", type=str, default=None,
                   help="Path to concept-erased UNet checkpoint (.pt).")
    p.add_argument("--target", type=str, required=True,
                   help="Target artist name (must match a CSV: datasets/test_<target>.csv).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results/artist_eval")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = os.path.join(args.output_dir, args.target)
    os.makedirs(out_dir, exist_ok=True)

    # Load artist prompts
    csv_path = f"datasets/test_{args.target}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    data = pd.read_csv(csv_path)
    prompts = data["prompt"].tolist()
    seeds = data["evaluation_seed"].tolist()
    print(f"Loaded {len(prompts)} prompts for artist '{args.target}'")

    # Generate images if not cached
    orig_dir = os.path.join(out_dir, "original")
    erased_dir = os.path.join(out_dir, "erased")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(erased_dir, exist_ok=True)

    need_generation = any(
        not os.path.exists(os.path.join(erased_dir, f"{i}.jpg"))
        for i in range(len(prompts))
    )

    if need_generation:
        print("Generating images...")
        base_pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id, torch_dtype=torch.float16
        ).to(args.gpu)
        erased_pipe = load_erased_model(args.model_id, args.ckpt_name, args.gpu)

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            orig_path = os.path.join(orig_dir, f"{i}.jpg")
            erased_path = os.path.join(erased_dir, f"{i}.jpg")

            if os.path.exists(orig_path) and os.path.exists(erased_path):
                print(f"[{i}] cached, skipping")
                continue

            print(f"[{i}] {prompt}  (seed={seed})")
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            erased_img = erased_pipe(prompt).images[0]

            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            orig_img = base_pipe(prompt).images[0]

            orig_img.save(orig_path)
            erased_img.save(erased_path)

        # Free GPU memory
        del base_pipe, erased_pipe
        torch.cuda.empty_cache()

    # CLIP evaluation
    print("\nCalculating CLIP scores...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    similarities_erased = []
    erasure_wins = []

    for i, prompt in enumerate(prompts):
        orig_img = Image.open(os.path.join(orig_dir, f"{i}.jpg")).convert("RGB")
        erased_img = Image.open(os.path.join(erased_dir, f"{i}.jpg")).convert("RGB")

        # Text features
        txt = clip_proc(text=prompt, return_tensors="pt", padding=True).to(device)
        txt_feat = clip_model.get_text_features(**txt)

        # Image features
        orig_inp = clip_proc(images=orig_img, return_tensors="pt").to(device)
        orig_feat = clip_model.get_image_features(**orig_inp)

        erased_inp = clip_proc(images=erased_img, return_tensors="pt").to(device)
        erased_feat = clip_model.get_image_features(**erased_inp)

        sim_orig = torch.nn.functional.cosine_similarity(txt_feat, orig_feat).item()
        sim_erased = torch.nn.functional.cosine_similarity(txt_feat, erased_feat).item()

        similarities_erased.append(sim_erased)
        # Score=1 if erased model is LESS aligned with the artist prompt (good)
        erasure_wins.append(1 if sim_erased < sim_orig else 0)

    avg_sim = float(np.mean(similarities_erased))
    avg_win = float(np.mean(erasure_wins))

    print(f"Average CLIP similarity (erased): {avg_sim:.4f}")
    print(f"Erasure win rate: {avg_win:.4f}  (fraction where erased < original)")

    tag = (
        os.path.basename(args.ckpt_name).split(".")[0]
        if args.ckpt_name
        else "original"
    )
    results = {
        "avg_similarity_erased": avg_sim,
        "erasure_win_rate": avg_win,
        "std_similarity": float(np.std(similarities_erased)),
        "std_win_rate": float(np.std(erasure_wins)),
        "num_prompts": len(prompts),
    }
    result_path = os.path.join(out_dir, f"clip_scores_{tag}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()