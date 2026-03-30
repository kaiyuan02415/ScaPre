# res acc
python benchmarking/object_erase.py \
    --target "your_target" \
    --baseline concept-prune \
    --removal_mode erase  \
    --ckpt_name "your_saved_checkpoint.pt"

# coco clip score
python benchmarking/eval_coco_clip.py \
    --prompt_file "datasets/coco_prompts.txt" \
    --ckpt_name "your_saved_checkpoint.pt" \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --target "your_target" 

python nudity_eval.py \
    --ckpt_name your_saved_checkpoint.pt \
    --eval_dataset i2p \
    --output_dir results/nudity_eval

python artist_erasure.py \
    --ckpt_name your_saved_checkpoint.pt \
    --target "your_target" \
    --output_dir results/artist_eval

