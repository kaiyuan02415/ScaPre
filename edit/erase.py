import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import pandas as pd 
import argparse
import requests
import os, glob, json
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import abc
import copy
from functools import reduce
import operator
import ast  # Note: ast.literal_eval is used in some parts of the code

eps = 1e-8

def view_images(images, num_rows=3, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (batch_size, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt,
    num_inference_steps = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    low_resource = False,
):
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    total_steps = len(model.scheduler.timesteps)
    print(f"Starting diffusion process with {total_steps} steps...")
    for i, t in enumerate(model.scheduler.timesteps, start=1):
        print(f"Diffusion progress: step {i}/{total_steps}")
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)

    # image, _ = model.run_safety_checker(image=image, device=model.device, dtype=text_embeddings.dtype)
  
    return image


def generate_for_text(ldm_stable, test_text, num_samples = 9, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text]*num_samples, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    return view_images(images)


def get_ratios(ldm_stable, prev_ratio, ratio_diff, max_ratio_gap, concepts, classes, num_samples=10, num_loops=3):
    seeds = np.random.randint(5000, size=5) 
    ratios = []
    for idx, concept in enumerate(concepts):
        if ratio_diff is not None:
            if ratio_diff[idx] < max_ratio_gap:
                print(f'Bypassing Concept {idx+1}')
                ratios.append(prev_ratio[idx])
                continue
        print(f"Processing concept {idx+1}/{len(concepts)}: {concept}")
        prompt = f'{concept}'
        probs_full = []
        test_prompts = [f'{class_}' for class_ in classes[idx]]
        with torch.no_grad():
            for seed_index, seed in enumerate(seeds, start=1):
                print(f"\tProcessing seed {seed_index}/{len(seeds)}: {seed}")
                g = torch.Generator(device='cpu')
                g.manual_seed(int(seed))
                images = ldm_stable(prompt, num_images_per_prompt=num_samples, num_inference_steps=20, generator=g).images

                inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
                
        ratios.append(torch.cat(probs_full).mean(axis=0))
#     male = float(probs[0][0])
    return ratios

# ─────────────────────────────────────────────────────────────
#  Entropy helper – compute per-channel information entropy and return retention factor e_i∈[0,1]
# ─────────────────────────────────────────────────────────────
def _compute_entropy_factor(W_old, concept_vecs,
                            num_samples=20, bins=20,
                            noise_sigma=0.01, eps=1e-8):
    """
    Input
    ----
    W_old        : (d_out, d_emb) original weight
    concept_vecs : list(tensor)   m concept token vectors  (d_emb,)
    Return
    ----
    e (tensor)   : (d_out,)       entropy retention factor, closer to 1 means more specialized
    """
    out_dim, _ = W_old.shape
    acts = []
    # 1) Generate diverse activation samples using concept vectors + noise
    for c in concept_vecs:
        for _ in range(num_samples):
            noise = torch.randn_like(c) * noise_sigma
            acts.append((W_old @ (c + noise)).unsqueeze(1))  # (d_out,1)
    acts = torch.cat(acts, dim=1)  # (d_out, S)
    # 2) Normalize to [0,1]
    min_v = acts.min(dim=1, keepdim=True).values
    max_v = acts.max(dim=1, keepdim=True).values
    acts = (acts - min_v) / (max_v - min_v + eps)
    # 3) Shannon entropy
    e = torch.zeros(out_dim, device=W_old.device)
    logK = torch.log(torch.tensor(float(bins), device=W_old.device))
    for i in range(out_dim):
        hist = torch.histc(acts[i], bins=bins, min=0.0, max=1.0)
        p = hist / (hist.sum() + eps)
        H = -(p * (p + eps).log()).sum()
        e[i] = 1 - H / logK          # Normalized: lower entropy -> closer to 1
    return e



# ─────────────────────────────────────────────────────────────
# MI-SoftMask: uses empty prompt embedding as negative examples
# ─────────────────────────────────────────────────────────────
def _compute_mi_softmask_emptyneg(W_old: torch.Tensor,
                                  c_vec:  torch.Tensor,
                                  empty_vec: torch.Tensor,
                                  num_pos: int     = 5,
                                  T: float         = 0.7,
                                  p: float         = 2.0,
                                  noise_sigma: float = 0.05,
                                  eps: float       = 1e-8) -> torch.Tensor:
    """
    Positive examples: c_vec + epsilon  (num_pos samples)
    Negative examples: empty_vec + epsilon  (num_pos samples)
    Returns: (d_out,1) SoftMask
    """
    device = W_old.device
    out_dim, _ = W_old.shape

    # 1) Sample positive / negative examples
    pos = c_vec.repeat(num_pos,1) \
          + noise_sigma * torch.randn(num_pos, c_vec.numel(), device=device)
    neg = empty_vec.repeat(num_pos,1) \
          + noise_sigma * torch.randn(num_pos, c_vec.numel(), device=device)
    samples = torch.cat([pos, neg], dim=0)           # (2*num_pos, d_emb)
    labels  = torch.cat([torch.ones(num_pos,  device=device),
                         torch.zeros(num_pos, device=device)])  # (2K,)

    # 2) Row activations
    acts = W_old @ samples.t()                       # (d_out, 2K)

    # 3) Row median binarization
    tau = acts.median(dim=1, keepdim=True).values
    Z   = (acts > tau).long()                       # (d_out,2K)

    # 4) Per-row mutual information
    K  = 2 * num_pos
    mi = torch.zeros(out_dim, device=device)
    for i in range(out_dim):
        z = Z[i]
        n11 = ((z==1)&(labels==1)).sum().float() + eps
        n10 = ((z==1)&(labels==0)).sum().float() + eps
        n01 = ((z==0)&(labels==1)).sum().float() + eps
        n00 = ((z==0)&(labels==0)).sum().float() + eps
        p11, p10 = n11/K, n10/K
        p01, p00 = n01/K, n00/K
        p1_, p0_ = p11+p10, p01+p00
        p_1, p_0 = p11+p01, p10+p00
        mi[i] = (p11*torch.log(p11/(p1_*p_1))
                +p10*torch.log(p10/(p1_*p_0))
                +p01*torch.log(p01/(p0_*p_1))
                +p00*torch.log(p00/(p0_*p_0)))

    # 5) Standardize & SoftMask
    mi_std = (mi - mi.mean()) / (mi.std() + eps)
    m = torch.sigmoid(mi_std / T).pow(p)
    return m.view(-1,1)

def smooth_svd_on_mat1(mat1: torch.Tensor,
                       C_stack: torch.Tensor,
                       W_old: torch.Tensor,
                       in_dim: int,
                       device: torch.device,
                       T_sigma: float = 0.5,
                       p_sigma: float = 2.0,
                       eps: float = 1e-8):
    """
    Update mat1 using Spectral Smooth Attenuation (SSA-SVD)
    mat1      : (d_out, d_in) -- UCE left-side accumulation matrix
    C_stack   : (d_in, m)     -- stacked concept matrix (res_ctx stacked)
    W_old     : (d_out, d_in) -- current layer original weight
    Returns   : updated mat1
    """
    # 1) SVD on C_stack
    U, S, _ = torch.linalg.svd(C_stack, full_matrices=False)   # U:(d_in,r)
    # 2) Compute attenuation weights
    tau = torch.median(S)
    w   = torch.sigmoid((S - tau)/(T_sigma + eps)).pow(p_sigma)  # (r,)
    # 3) Smooth projection matrix P = I - U diag(w) U^T
    U_w = U * w.unsqueeze(0)                                   # (d_in,r)
    P   = torch.eye(in_dim, device=device) - U_w @ U.T         # (d_in,d_in)
    # 4) Project residual (V_res = W_old * P * C_stack)
    C_res = P @ C_stack                                        # (d_in,m)
    V_res = W_old @ C_res                                      # (d_out,m)
    # 5) Add -eff * (V_res C_res^T) to mat1 -- done in the outer loop
    return V_res, C_res


import copy, ast, torch

# def edit_model(ldm_stable,
#                old_text_, new_text_, retain_text_,
#                add=False, layers_to_edit=None,
#                lamb=0.1, erase_scale=0.1, preserve_scale=0.1,
#                with_to_k=True, technique='tensor',
#                svd=False,
#                p=2.0,                    # row soft-mask power exponent
#                res_scale=None,           # residual SVD step strength (None -> erase_scale)
#                alpha_min=0.1,
#                entropy_samples=50,       # number of samples per concept
#                entropy_bins=20,          # histogram bin count
#                noise_sigma=0.01,
#                T_sigma=1,
#                p_sigma=1):        
#     """
#     Weighted-UCE (soft row weights) + residual low-rank (svd) + adaptive rollback
#     ─────────────────────────────────────────────────────────────────────────────
#     . saliency m = (|Wc|/max)^p  (continuous mask, default p=2)
#     · svd=True  → ΔW_res = –res_scale·(W C_res)C_resᵀ
#     . Adaptive rollback before final write-back:
#         ν = ||W_new–W_old||/||W_old||,  
#         αₗ = max(exp(−κ·ν), α_min),  
#         W_final = αₗ·W_new + (1−αₗ)·W_old
#     """
    
#     ### collect all the cross attns modules
#     sub_nets = ldm_stable.unet.named_children()
#     ca_layers = []
#     for net in sub_nets:
#         if 'up' in net[0] or 'down' in net[0]:
#             for block in net[1]:
#                 if 'Cross' in block.__class__.__name__ :
#                     for attn in block.attentions:
#                         for transformer in attn.transformer_blocks:
#                             ca_layers.append(transformer.attn2)
#         if 'mid' in net[0]:
#             for attn in net[1].attentions:
#                 for transformer in attn.transformer_blocks:
#                     ca_layers.append(transformer.attn2)

#     ### get the value and key modules
#     projection_matrices = [l.to_v for l in ca_layers]
#     og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
#     if with_to_k:
#         projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
#         og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

#     ## reset the parameters
#     num_ca_clip_layers = len(ca_layers)
#     for idx_, l in enumerate(ca_layers):
#         l.to_v = copy.deepcopy(og_matrices[idx_])
#         projection_matrices[idx_] = l.to_v
#         if with_to_k:
#             l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
#             projection_matrices[num_ca_clip_layers + idx_] = l.to_k

#     ### check the layers to edit
#     layers_to_edit = ast.literal_eval(layers_to_edit) if isinstance(layers_to_edit, str) else layers_to_edit
#     lamb = ast.literal_eval(lamb) if isinstance(lamb, str) else lamb

#     ### Format the edits
#     old_texts, new_texts = [], []
#     for o, n in zip(old_text_, new_text_):
#         old_texts.append(o)
#         new_texts.append(' ' if n == '' else n)
#     ret_texts = [''] if retain_text_ is None else retain_text_

#     # === Pre-extract all concept vectors (once) ====================================
#     tok, enc, device = ldm_stable.tokenizer, ldm_stable.text_encoder, ldm_stable.device
#     text_inp = tok(old_texts, padding="max_length",
#                    max_length=tok.model_max_length,
#                    truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         embeds_all = enc(text_inp.input_ids.to(device))[0]
#     concept_vecs = [emb[mask.sum()-2] for emb, mask in zip(embeds_all, text_inp.attention_mask)]


#     #  Empty prompt vector -- executed only once
#     with torch.no_grad():
#         blank = tok([""], padding="max_length",
#                     max_length=tok.model_max_length,
#                     truncation=True, return_tensors="pt")
#         blank_emb = enc(blank.input_ids.to(device))[0]
#     empty_vec = blank_emb[0, 1, :].detach()


#     C_full = torch.stack(concept_vecs, 1)          # (d,m)
#     CCt    = C_full @ C_full.t()                   # (d,d)
#     m_concepts = C_full.shape[1]
#     # ===============================================================

#     print(old_texts, new_texts)
#     total = len(projection_matrices)          # = 32
    
#     ######################## START ERASING ##########################
#     for layer_num, ca in enumerate(ca_layers, 1):
#         if layers_to_edit is not None and layer_num-1 not in layers_to_edit:
#             continue
#         print(f"Editing layer {layer_num}/{len(ca_layers)}")
#         with torch.no_grad():
#             W_old = ca.to_v.weight.data.clone()              # save old weights
#             out_dim, in_dim = W_old.shape

#             mat1 = lamb * W_old.clone()
#             mat2 = lamb * torch.eye(in_dim, device=device) + CCt

#             res_vals, res_ctx = [], []

#             # ---------- ERASE ----------
#             for idx, (ot, nt, c_vec) in enumerate(zip(old_texts, new_texts, concept_vecs), 1):
#                 print(f"\tEditing text pair {idx}/{len(old_texts)}")
#                 tinp = tok([ot, nt], padding="max_length",
#                            max_length=tok.model_max_length,
#                            truncation=True, return_tensors="pt")
#                 emb  = enc(tinp.input_ids.to(device))[0]
#                 f_o  = tinp.attention_mask[0].sum().item() - 2
#                 f_n  = tinp.attention_mask[1].sum().item() - 2
#                 far  = max(f_o, f_n)
#                 old_emb = emb[0][f_o: len(emb[0])-max(0,far-f_o)]
#                 new_emb = emb[1][f_n: len(emb[1])-max(0,far-f_n)]
#                 context = old_emb.detach()

#                 # value computation
#                 vals=[]
#                 for layer in ca_layers:
#                     if technique == 'tensor':
#                         u = layer.to_v(context).detach(); u = u/u.norm()
#                         v = layer.to_v(new_emb).detach()
#                         vals.append((v - (u*v).sum()*u).detach())
#                     else:
#                         vals.append(layer.to_v(new_emb).detach())

#                 ctx_v  = context.view(context.size(0), context.size(1), 1)
#                 ctx_vT = context.view(context.size(0), 1, context.size(1))
#                 val_v  = vals[layer_num-1].view(vals[layer_num-1].size(0),
#                                                 vals[layer_num-1].size(1), 1)
#                 for_mat1 = (val_v @ ctx_vT).sum(0)
#                 for_mat2 = (ctx_v  @ ctx_vT).sum(0)

#                 # sal = torch.abs(W_old @ (c_vec / (c_vec.norm()+1e-8)))
#                 # row_w = (sal / (sal.max()+1e-8)).pow(p).view(-1,1)

#                 # -- MI-SoftMask replaces row-norm SoftMask --
#                 row_w = _compute_mi_softmask_emptyneg(
#                     W_old     = W_old,
#                     c_vec     = c_vec,
#                     empty_vec = empty_vec,
#                     num_pos   = 5,
#                     T         = 0.7,
#                     p         = p,
#                     noise_sigma = noise_sigma
#                 )

#                 # -- Compute matrix update --
#                 mat1 += erase_scale * (for_mat1 * row_w)
#                 mat2 += erase_scale * for_mat2

#                 res_vals.append((W_old @ c_vec).detach())
#                 res_ctx.append(c_vec.detach())

#             # ---------- PRESERVE ----------
#             # for pt in ret_texts:
#             #     if pt == '': continue
#             #     pinp = tok([pt, pt], padding="max_length",
#             #                max_length=tok.model_max_length,
#             #                truncation=True, return_tensors="pt")
#             #     pemb = enc(pinp.input_ids.to(device))[0]
#             #     ctx, nctx = pemb
#             #     ctx = ctx.detach()
#             #     val = ca.to_v(nctx).detach()
#             #     ctx_v  = ctx.view(ctx.size(0), ctx.size(1), 1)
#             #     ctx_vT = ctx.view(ctx.size(0), 1, ctx.size(1))
#             #     val_v  = val.view(val.size(0), val.size(1), 1)
#             #     mat1 += preserve_scale * ((val_v @ ctx_vT).sum(0))
#             #     mat2 += preserve_scale * ((ctx_v  @ ctx_vT).sum(0))

#             # ---------- residual SVD  (SSA-SVD on mat1) ----------
#             if svd:
#                 eff = erase_scale if res_scale is None else res_scale

#                 # (1) Stack res_ctx into (d_in, m)
#                 C_stack = torch.stack(res_ctx, 1)          # (d_in, m)

#                 # (2) Call smooth SVD function, get V_res, C_res
#                 V_res, C_res = smooth_svd_on_mat1(
#                     mat1     = mat1,
#                     C_stack  = C_stack,
#                     W_old    = W_old,
#                     in_dim   = in_dim,
#                     device   = device,
#                     T_sigma  = T_sigma,
#                     p_sigma  = p_sigma
#                 )

#                 # (3) Same as original: mat1 += -eff * (V_res C_res^T)
#                 mat1 += -eff * (V_res @ C_res.t())

#             W_new = mat1 @ torch.inverse(mat2)
#             # ---------- solve & adaptive rollback ----------
#             # -------------- Entropy-driven rollback --------------
#             e_i = _compute_entropy_factor(
#                 W_old, concept_vecs,
#                 num_samples=entropy_samples,
#                 bins=entropy_bins,
#                 noise_sigma=noise_sigma
#             )                                   # (out_dim,)
#             alpha = alpha_min + (1 - alpha_min) * e_i
#             W_final = alpha.view(-1,1) * W_new + (1-alpha).view(-1,1) * W_old
#             ca.to_v.weight.data.copy_(W_final)

            
#     print(f'[edit_model] done | p={p} | svd={svd} | entropy rollback samples={entropy_samples} | alpha_min={alpha_min}')
#     return ldm_stable

import ast, copy, torch

def edit_model(ldm_stable,
               old_text_, new_text_, retain_text_,
               add=False, layers_to_edit=None,
               lamb=0.1, erase_scale=0.1, preserve_scale=0.1,
               with_to_k=True, technique='tensor',
               svd=False,
               p=2.0,                    # row soft-mask power exponent
               res_scale=None,           # residual SVD step strength (None -> erase_scale)
               alpha_min=0.1,
               entropy_samples=50,       # number of samples per concept
               entropy_bins=20,          # histogram bin count
               noise_sigma=0.01,
               T_sigma=1,
               p_sigma=1):
    """
    Weighted-UCE (soft row weights) + residual low-rank (svd) + adaptive rollback
    ─────────────────────────────────────────────────────────────────────────────
    Added support for to_k; other logic remains unchanged
    """

    # === 0. Collect cross-attention modules =================================================
    sub_nets   = ldm_stable.unet.named_children()
    ca_layers  = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    # === 1. Backup value / key weights in advance ===============================================
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices         = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices += [l.to_k for l in ca_layers]
        og_matrices         += [copy.deepcopy(l.to_k) for l in ca_layers]

    # === 2. Reset weights (ensure idempotent repeated runs) ===============================================
    num_ca = len(ca_layers)
    for idx, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx])
        projection_matrices[idx] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca + idx])
            projection_matrices[num_ca + idx] = l.to_k

    # === 3. Parse parameters / text vectors ========================================================
    layers_to_edit = ast.literal_eval(layers_to_edit) if isinstance(layers_to_edit, str) else layers_to_edit
    lamb = ast.literal_eval(lamb) if isinstance(lamb, str) else lamb

    old_texts, new_texts = [], []
    for o, n in zip(old_text_, new_text_):
        old_texts.append(o)
        new_texts.append(' ' if n == '' else n)
    ret_texts = [''] if retain_text_ is None else retain_text_

    tok, enc, device = ldm_stable.tokenizer, ldm_stable.text_encoder, ldm_stable.device
    text_inp  = tok(old_texts, padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds_all = enc(text_inp.input_ids.to(device))[0]
    concept_vecs = [emb[mask.sum()-2] for emb, mask in zip(embeds_all, text_inp.attention_mask)]

    with torch.no_grad():
        blank = tok([""], padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt")
        blank_emb = enc(blank.input_ids.to(device))[0]
    empty_vec = blank_emb[0, 1, :].detach()

    C_full = torch.stack(concept_vecs, 1)          # (d, m)
    CCt    = C_full @ C_full.t()                   # (d, d)

    # === 4-A. Edit 16 to_v layers ==========================================================
    for layer_num, ca in enumerate(ca_layers, 1):
        if layers_to_edit is not None and layer_num-1 not in layers_to_edit:
            continue
        print(f"Editing to_v   {layer_num}/{len(ca_layers)}")
        with torch.no_grad():
            W_old = ca.to_v.weight.data.clone()
            out_dim, in_dim = W_old.shape
            mat1 = lamb * W_old.clone()
            mat2 = lamb * torch.eye(in_dim, device=device) + CCt
            res_vals, res_ctx = [], []

            # ---------- ERASE ----------
            for idx, (ot, nt, c_vec) in enumerate(zip(old_texts, new_texts, concept_vecs), 1):
                tinp = tok([ot, nt], padding="max_length",
                           max_length=tok.model_max_length,
                           truncation=True, return_tensors="pt")
                emb  = enc(tinp.input_ids.to(device))[0]
                f_o  = tinp.attention_mask[0].sum().item() - 2
                f_n  = tinp.attention_mask[1].sum().item() - 2
                far  = max(f_o, f_n)
                old_emb = emb[0][f_o: len(emb[0])-max(0,far-f_o)]
                new_emb = emb[1][f_n: len(emb[1])-max(0,far-f_n)]
                context = old_emb.detach()

                vals=[]
                for layer in ca_layers:
                    if technique == 'tensor':
                        u = layer.to_v(context).detach(); u = u/u.norm()
                        v = layer.to_v(new_emb).detach()
                        vals.append((v - (u*v).sum()*u).detach())
                    else:
                        vals.append(layer.to_v(new_emb).detach())

                ctx_v  = context.view(context.size(0), context.size(1), 1)
                ctx_vT = context.view(context.size(0), 1, context.size(1))
                val_v  = vals[layer_num-1].view(vals[layer_num-1].size(0),
                                                vals[layer_num-1].size(1), 1)
                for_mat1 = (val_v @ ctx_vT).sum(0)
                for_mat2 = (ctx_v  @ ctx_vT).sum(0)

                row_w = _compute_mi_softmask_emptyneg(
                    W_old     = W_old,
                    c_vec     = c_vec,
                    empty_vec = empty_vec,
                    num_pos   = 5,
                    T         = 0.7,
                    p         = p,
                    noise_sigma = noise_sigma
                )
                mat1 += erase_scale * (for_mat1 * row_w)
                mat2 += erase_scale * for_mat2

                res_vals.append((W_old @ c_vec).detach())
                res_ctx.append(c_vec.detach())

            if svd:
                eff = erase_scale if res_scale is None else res_scale
                C_stack = torch.stack(res_ctx, 1)
                V_res, C_res = smooth_svd_on_mat1(
                    mat1     = mat1,
                    C_stack  = C_stack,
                    W_old    = W_old,
                    in_dim   = in_dim,
                    device   = device,
                    T_sigma  = T_sigma,
                    p_sigma  = p_sigma
                )
                mat1 += -eff * (V_res @ C_res.t())

            W_new = mat1 @ torch.inverse(mat2)
            e_i = _compute_entropy_factor(
                W_old, concept_vecs,
                num_samples=entropy_samples,
                bins=entropy_bins,
                noise_sigma=noise_sigma
            )
            alpha = alpha_min + (1 - alpha_min) * e_i
            W_final = alpha.view(-1,1) * W_new + (1-alpha).view(-1,1) * W_old
            ca.to_v.weight.data.copy_(W_final)

    # === 4-B. Additionally edit 16 to_k layers (only when with_to_k=True) ===============================
    if with_to_k:
        for layer_num, ca in enumerate(ca_layers, 1):
            if layers_to_edit is not None and layer_num-1 not in layers_to_edit:
                continue
            print(f"Editing to_k   {layer_num}/{len(ca_layers)}")
            with torch.no_grad():
                W_old = ca.to_k.weight.data.clone()
                out_dim, in_dim = W_old.shape
                mat1 = lamb * W_old.clone()
                mat2 = lamb * torch.eye(in_dim, device=device) + CCt
                res_vals, res_ctx = [], []

                for idx, (ot, nt, c_vec) in enumerate(zip(old_texts, new_texts, concept_vecs), 1):
                    tinp = tok([ot, nt], padding="max_length",
                               max_length=tok.model_max_length,
                               truncation=True, return_tensors="pt")
                    emb  = enc(tinp.input_ids.to(device))[0]
                    f_o  = tinp.attention_mask[0].sum().item() - 2
                    f_n  = tinp.attention_mask[1].sum().item() - 2
                    far  = max(f_o, f_n)
                    old_emb = emb[0][f_o: len(emb[0])-max(0,far-f_o)]
                    new_emb = emb[1][f_n: len(emb[1])-max(0,far-f_n)]
                    context = old_emb.detach()

                    vals=[]
                    for layer in ca_layers:
                        if technique == 'tensor':
                            u = layer.to_k(context).detach(); u = u/u.norm()
                            v = layer.to_k(new_emb).detach()
                            vals.append((v - (u*v).sum()*u).detach())
                        else:
                            vals.append(layer.to_k(new_emb).detach())

                    ctx_v  = context.view(context.size(0), context.size(1), 1)
                    ctx_vT = context.view(context.size(0), 1, context.size(1))
                    val_v  = vals[layer_num-1].view(vals[layer_num-1].size(0),
                                                    vals[layer_num-1].size(1), 1)
                    for_mat1 = (val_v @ ctx_vT).sum(0)
                    for_mat2 = (ctx_v  @ ctx_vT).sum(0)

                    row_w = _compute_mi_softmask_emptyneg(
                        W_old     = W_old,
                        c_vec     = c_vec,
                        empty_vec = empty_vec,
                        num_pos   = 5,
                        T         = 0.7,
                        p         = p,
                        noise_sigma = noise_sigma
                    )
                    mat1 += erase_scale * (for_mat1 * row_w)
                    mat2 += erase_scale * for_mat2

                    res_vals.append((W_old @ c_vec).detach())
                    res_ctx.append(c_vec.detach())

                if svd:
                    eff = erase_scale if res_scale is None else res_scale
                    C_stack = torch.stack(res_ctx, 1)
                    V_res, C_res = smooth_svd_on_mat1(
                        mat1     = mat1,
                        C_stack  = C_stack,
                        W_old    = W_old,
                        in_dim   = in_dim,
                        device   = device,
                        T_sigma  = T_sigma,
                        p_sigma  = p_sigma
                    )
                    mat1 += -eff * (V_res @ C_res.t())

                W_new = mat1 @ torch.inverse(mat2)
                e_i = _compute_entropy_factor(
                    W_old, concept_vecs,
                    num_samples=entropy_samples,
                    bins=entropy_bins,
                    noise_sigma=noise_sigma
                )
                alpha = alpha_min + (1 - alpha_min) * e_i
                W_final = alpha.view(-1,1) * W_new + (1-alpha).view(-1,1) * W_old
                ca.to_k.weight.data.copy_(W_final)

    print(f'[edit_model] done | p={p} | svd={svd} | entropy rollback samples={entropy_samples} | alpha_min={alpha_min}')
    return ldm_stable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='TrainUSD',
                    description='Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concepts', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--guided_concepts', help='Concepts to guide the erased concepts towards', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=None)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, required=True)
    parser.add_argument('--add_prompts', help='option to add additional prompts', type=bool, required=False, default=False)

    # NEW parameters
    parser.add_argument('--svd', help='whether to enable residual low-rank step', 
                    action='store_true', default=False)
    parser.add_argument('--p',          type=float, default=2.0,
                        help='exponent for soft row-weight (default 2)')
    parser.add_argument('--res_scale',  type=float, default=None,
                        help='scale for residual SVD step; None → use erase_scale')
    # parser.add_argument('--kappa',     type=float, default=10.0,
    #                     help='adaptive rollback sensitivity (larger -> more rollback)')
    parser.add_argument('--alpha_min', type=float, default=0.1,
                        help='minimum layer rollback factor')
    
    parser.add_argument(
    "--entropy_samples",
    type=int,
    default=20,
    help="number of pseudo-samples generated/sampled per concept for channel entropy estimation"
    )
    parser.add_argument(
        "--entropy_bins",
        type=int,
        default=20,
        help="histogram bin count, controls entropy resolution (default: 20)"
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.01,
        help="std of Gaussian noise added to concept vectors for sample diversity (default: 0.01)"
    )

    parser.add_argument('--T_sigma', type=float, default=1,
                    help='SVD spectral temperature (larger -> smoother)')
    parser.add_argument('--p_sigma', type=float, default=1,
                    help='SVD spectral power exponent (larger -> harder decay)')
    
    args = parser.parse_args()
    technique = args.technique
    device = f'cuda:{args.device}'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    add_prompts = args.add_prompts
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    preserve_number = args.preserve_number
    concepts = args.concepts.split(',')
    concepts = [con.strip() for con in concepts]
    concept_type = args.concept_type
    print_text = ''
    for txt in concepts:
        print_text += txt.lower() + '_'
    print_text = print_text[:-1]
    # PROMPT CLEANING
    if concepts[0] == 'allartist':
        concepts = ["Kelly Mckernan", "Thomas Kinkade", "Pablo Picasso", "Tyler Edlin", "Kilian Eng"]
    if concepts[0] == 'i2g':
        concepts = ['hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide', 'sexual', 'nudity', 'bodily fluids', 'blood']
    if concepts[0] == '10artists':
        concepts = ["Asger Jorn", "Eric Fischl", "Johannes Vermeer", "Apollinary Vasnetsov", "Naoki Urasawa", "Nicolas Mignard", "John Whitcomb", "John Constable", "Warwick Globe", "Albert Marquet"]
    if concepts[0] == 'imagenette':
        concepts = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']

    if 'artists' in concepts[0]:
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = int(concepts[0].replace('artists', ''))
        concepts = random.sample(artists, number) 

    old_texts = []
    
    additional_prompts = []
    if concept_type == 'art':
        additional_prompts.append('painting by {concept}')
        additional_prompts.append('art by {concept}')
        additional_prompts.append('artwork by {concept}')
        additional_prompts.append('picture by {concept}')
        additional_prompts.append('style of {concept}')
    elif concept_type=='object':
        additional_prompts.append('image of {concept}')
        additional_prompts.append('photo of {concept}')
        additional_prompts.append('portrait of {concept}')
        additional_prompts.append('picture of {concept}')
        additional_prompts.append('painting of {concept}')  
    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))
        length = 1 + len(additional_prompts)
        concepts_.extend([concept] * length)
    
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text += f'-towards_uncond'
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text += f'-towards_{guided_concepts[0]}'
        else:
            new_texts = [[con] * length for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts)
            print_text += f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text += f'-{t}'
            
    assert len(new_texts) == len(old_texts)
    
    
    if preserve_concepts is None:
        if concept_type == 'art':
            df = pd.read_csv('data/artists1734_prompts.csv')

            retain_texts = list(df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text += f'-preserving_{preserve_number}artists'
                preserve_concepts = random.sample(preserve_concepts, preserve_number)
        else:
            preserve_concepts = []

    retain_texts = [''] + preserve_concepts   
    if len(retain_texts) > 1:
        print_text += f'-preserve_true'     
    else:
        print_text += f'-preserve_false'
    if preserve_scale is None:
        preserve_scale = max(0.1, 1/len(retain_texts))

    sd14 = "CompVis/stable-diffusion-v1-4"
    sd15 = "runwayml/stable-diffusion-v1-5"
    sd21 = 'stabilityai/stable-diffusion-2-1-base'

    if args.base == '1.4':
        model_version = sd14
    elif args.base == '1.5':
        model_version = sd15
    elif args.base == '2.1':
        model_version = sd21
    else:
        model_version = sd14
    print(f"Model selection: using base version {args.base} ({model_version})")
    print("Loading Stable Diffusion pipeline...")
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    print("Stable Diffusion model loaded successfully.")
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text = print_text.lower()
    print(print_text)
    print("Starting model editing...")
    ldm_stable = edit_model(ldm_stable=ldm_stable, old_text_=old_texts, new_text_=new_texts, 
                            with_to_k=True,
                            add=False, retain_text_=retain_texts, lamb=0.5, 
                            erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique, 
                            svd=args.svd, p=args.p, res_scale=args.res_scale, 
                            alpha_min=args.alpha_min,
                            entropy_samples     = args.entropy_samples,
                            entropy_bins        = args.entropy_bins,
                            noise_sigma         = args.noise_sigma,
                            T_sigma=args.T_sigma,
                            p_sigma=args.p_sigma)
    
    print("Model editing completed.")
    
    torch.save(ldm_stable.unet.state_dict(), f'models/your_saved_model.pt')