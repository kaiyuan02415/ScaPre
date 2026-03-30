"""
Object Erasure Evaluation using ResNet-50 Classification.

Tests whether a concept-erased model still generates recognizable
instances of the erased object class (using ImageNet-pretrained ResNet-50).
Also supports 'keep' mode: checks that non-erased classes are preserved.

Usage (erase mode - lower accuracy = better erasure):
    python object_erase.py \
        --ckpt_name path/to/pruned_unet.pt \
        --target "church" \
        --mode erase \
        --output_dir results/object_erase

Usage (keep mode - higher accuracy = better preservation):
    python object_erase.py \
        --ckpt_name path/to/pruned_unet.pt \
        --target "church" \
        --mode keep \
        --output_dir results/object_keep
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.models import resnet50, ResNet50_Weights


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
# Dataset helpers
# ---------------------------------------------------------------------------
def load_imagenette_prompts(csv_path: str, target: str, mode: str):
    """
    Load prompts from imagenette CSV.
    - mode='erase': only prompts matching the target class
    - mode='keep':  only prompts NOT matching the target class
    Returns list of (prompt, seed, label) tuples.
    """
    df = pd.read_csv(csv_path)
    label_col = "class" if "class" in df.columns else "label_str"
    results = []
    for _, row in df.iterrows():
        label = row[label_col].lower()
        match = (label == target.lower())
        if (mode == "erase" and match) or (mode == "keep" and not match):
            results.append((row["prompt"], int(row["evaluation_seed"]), label))
    print(f"Loaded {len(results)} prompts (mode={mode}, target={target})")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = ArgumentParser(description=__doc__,
                       formatter_class=ArgumentParser.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--ckpt_name", type=str, default=None,
                   help="Path to concept-erased UNet checkpoint (.pt).")
    p.add_argument("--target", type=str, required=True,
                   help="Target concept to erase (must match a class in the CSV).")
    p.add_argument("--mode", type=str, required=True, choices=["erase", "keep"],
                   help="'erase': eval on target class; 'keep': eval on other classes.")
    p.add_argument("--dataset_csv", type=str, default="datasets/imagenette.csv")
    p.add_argument("--max_prompts", type=int, default=130)
    p.add_argument("--output_dir", type=str, default="results/object_eval")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = os.path.join(args.output_dir, args.target, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    samples = load_imagenette_prompts(args.dataset_csv, args.target, args.mode)
    samples = samples[: args.max_prompts]

    # Load erased model
    pipe = load_erased_model(args.model_id, args.ckpt_name, args.gpu)

    # ResNet-50 classifier
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights).to(args.gpu).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    correct = 0
    total = len(samples)

    for i, (prompt, seed, label) in enumerate(samples):
        print(f"[{i}/{total}] prompt='{prompt}', seed={seed}, label={label}")

        img_path = os.path.join(out_dir, f"img_{i}.png")

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            torch.manual_seed(seed)
            np.random.seed(seed)
            image = pipe(prompt).images[0]
            image.save(img_path)

        # Classify
        inp = preprocess(image).unsqueeze(0).to(args.gpu)
        with torch.no_grad():
            logits = classifier(inp)
        top1_idx = logits.argmax(dim=-1).item()
        pred = categories[top1_idx].lower()
        hit = label in pred or pred in label
        if hit:
            correct += 1
        print(f"  pred='{pred}', match={hit}")

    acc = correct / max(total, 1)
    print(f"\nObject detected in {correct}/{total} images  (accuracy={acc:.4f})")
    if args.mode == "erase":
        print("(Lower is better for erasure.)")
    else:
        print("(Higher is better for preservation.)")

    tag = (
        os.path.basename(args.ckpt_name).split(".")[0]
        if args.ckpt_name
        else "original"
    )
    results = {"accuracy": acc, "correct": correct, "total": total, "mode": args.mode}
    result_path = os.path.join(out_dir, f"results_{tag}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()