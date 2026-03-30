import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import argparse
import requests
import os, glob, json
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import abc
import copy
from functools import reduce
import operator
import ast

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
        ["" ] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    total_steps = len(model.scheduler.timesteps)
    for i, t in enumerate(model.scheduler.timesteps, start=1):
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)
    image = latent2image(model.vae, latents)
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
                ratios.append(prev_ratio[idx]); continue
        prompt = f'{concept}'
        probs_full = []
        test_prompts = [f'{class_}' for class_ in classes[idx]]
        with torch.no_grad():
            for seed in np.random.randint(5000, size=5):
                g = torch.Generator(device='cpu'); g.manual_seed(int(seed))
                images = ldm_stable(prompt, num_images_per_prompt=num_samples, num_inference_steps=20, generator=g).images
                inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
        ratios.append(torch.cat(probs_full).mean(axis=0))
    return ratios

# ─────────────────────────────────────────────────────────────
#  Entropy helper
# ─────────────────────────────────────────────────────────────
def _compute_entropy_factor(W_old, concept_vecs,
                            num_samples=20, bins=20,
                            noise_sigma=0.01, eps=1e-8):
    out_dim, _ = W_old.shape
    acts = []
    for c in concept_vecs:
        for _ in range(num_samples):
            noise = torch.randn_like(c) * noise_sigma
            acts.append((W_old @ (c + noise)).unsqueeze(1))
    acts = torch.cat(acts, dim=1)
    min_v = acts.min(dim=1, keepdim=True).values
    max_v = acts.max(dim=1, keepdim=True).values
    acts = (acts - min_v) / (max_v - min_v + eps)
    e = torch.zeros(out_dim, device=W_old.device)
    logK = torch.log(torch.tensor(float(bins), device=W_old.device))
    for i in range(out_dim):
        hist = torch.histc(acts[i], bins=bins, min=0.0, max=1.0)
        p = hist / (hist.sum() + eps)
        H = -(p * (p + eps).log()).sum()
        e[i] = 1 - H / logK
    return e

# ─────────────────────────────────────────────────────────────
# MI-SoftMask
# ─────────────────────────────────────────────────────────────
def _compute_mi_softmask_emptyneg(W_old: torch.Tensor,
                                  c_vec:  torch.Tensor,
                                  empty_vec: torch.Tensor,
                                  num_pos: int     = 5,
                                  T: float         = 0.7,
                                  p: float         = 2.0,
                                  noise_sigma: float = 0.05,
                                  eps: float       = 1e-8) -> torch.Tensor:
    device = W_old.device
    out_dim, _ = W_old.shape
    pos = c_vec.repeat(num_pos,1) + noise_sigma * torch.randn(num_pos, c_vec.numel(), device=device)
    neg = empty_vec.repeat(num_pos,1) + noise_sigma * torch.randn(num_pos, c_vec.numel(), device=device)
    samples = torch.cat([pos, neg], dim=0)
    labels  = torch.cat([torch.ones(num_pos,device=device), torch.zeros(num_pos,device=device)])
    acts = W_old @ samples.t()
    tau = acts.median(dim=1, keepdim=True).values
    Z   = (acts > tau).long()
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
    mi_std = (mi - mi.mean()) / (mi.std() + eps)
    m = torch.sigmoid(mi_std / T).pow(p)
    return m.view(-1,1)

# ─────────────────────────────────────────────────────────────
# ASED spectral regularizer
# ─────────────────────────────────────────────────────────────
def _ased_regularizer_from_C(C_full: torch.Tensor,
                             T_sigma: float = 1.0,
                             p_sigma: float = 1.0,
                             strength: float = 1.0,
                             eps: float = 1e-8) -> torch.Tensor:
    device = C_full.device
    U, S, _ = torch.linalg.svd(C_full, full_matrices=False)
    if S.numel() == 0:
        return torch.zeros(C_full.size(0), C_full.size(0), device=device)
    med = torch.median(S)
    gate = (1.0 / (1.0 + torch.exp(-(S - med) / (T_sigma + eps)))) ** p_sigma
    reg_diag = (1.0 - gate) * strength
    R = U @ torch.diag(reg_diag) @ U.t()
    return R

# ─────────────────────────────────────────────────────────────
# EGBR: row-level geometric safety valve
# ─────────────────────────────────────────────────────────────
def egbr_row(W0: torch.Tensor, Wstar: torch.Tensor, H_row: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    d_out = W0.size(0)
    Hmax = H_row.max().clamp_min(1e-6)
    alpha = 1 - torch.exp(-H_row / Hmax)  # [0,1]
    W0n = torch.norm(W0, dim=1, keepdim=True).clamp_min(eps)
    Wsn = torch.norm(Wstar, dim=1, keepdim=True).clamp_min(eps)
    u0 = W0 / W0n
    us = Wstar / Wsn
    cos = torch.sum(u0 * us, dim=1, keepdim=True)
    beta = alpha.view(-1,1) * torch.clamp(cos, min=0.0)
    u_tilde = (1 - beta) * u0 + beta * us
    u_tilde = u_tilde / torch.norm(u_tilde, dim=1, keepdim=True).clamp_min(eps)
    e0 = (W0n ** 2); es = (Wsn ** 2)
    enew = torch.exp((1 - alpha).view(-1,1) * torch.log(e0) + alpha.view(-1,1) * torch.log(es))
    Wnew = torch.sqrt(enew) * u_tilde
    return Wnew

# ─────────────────────────────────────────────────────────────
# NEW: Bures-Prox (true Bures proximal, 2-3 iterations usually sufficient)
# ─────────────────────────────────────────────────────────────
def bures_prox_row(W0: torch.Tensor, We: torch.Tensor,
                   mu: float, lam_prox: float,
                   iters: int = 2, eps: float = 1e-8) -> torch.Tensor:
    d_out, d_in = W0.shape
    a_norm = torch.norm(We, dim=1, keepdim=True).clamp_min(eps)     # ||a||
    u_a    = We / a_norm
    r0     = torch.norm(W0, dim=1, keepdim=True).clamp_min(eps)     # ||w0||
    u0     = W0 / r0

    u = u_a.clone()
    r = a_norm.clone()

    for _ in range(max(1, iters)):
        kappa = torch.clamp((u * u0).sum(dim=1, keepdim=True), min=0.0)
        r = (lam_prox * a_norm + mu * r0 * kappa) / (lam_prox + mu)
        num = lam_prox * a_norm * u_a + mu * r0 * kappa * u0
        u = num / torch.norm(num, dim=1, keepdim=True).clamp_min(eps)

    W = r * u
    return W

# ─────────────────────────────────────────────────────────────
# Core edit function (structure preserved, forgetting term replaced with "subspace projection-writeback + row weighting" + fused UCE accumulation)
# ─────────────────────────────────────────────────────────────
def edit_model(ldm_stable,
               old_text_, new_text_, retain_text_,
               add=False, layers_to_edit=None,
               lamb=0.1, erase_scale=0.1, preserve_scale=0.1,
               with_to_k=True, technique='tensor',
               svd=False,
               p=2.0,
               res_scale=None,
               alpha_min=0.1,
               entropy_samples=50,
               entropy_bins=20,
               noise_sigma=0.01,
               T_sigma=1,
               p_sigma=1,
               enable_ased=False,
               enable_egbr=False,
               bures_mu_from_entropy=True,
               use_mi_softmask=True,
               bures_iters=2):

    # === 0. Collect cross-attention modules
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

    # === 1. Backup to_v / to_k
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices         = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices += [l.to_k for l in ca_layers]
        og_matrices         += [copy.deepcopy(l.to_k) for l in ca_layers]

    # === 2. Reset weights (idempotent)
    num_ca = len(ca_layers)
    for idx, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx])
        projection_matrices[idx] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca + idx])
            projection_matrices[num_ca + idx] = l.to_k

    # === 3. Parse parameters / text vectors
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

    C_full = torch.stack(concept_vecs, 1)          # (d_in, m)
    CCt    = C_full @ C_full.t()                   # (d_in, d_in)
    # Concept subspace projection Pi_C
    CtC = C_full.t() @ C_full
    m_dim = CtC.size(0)
    PiC = C_full @ torch.linalg.inv(CtC + eps * torch.eye(m_dim, device=C_full.device)) @ C_full.t()

    # === 4-A. Edit to_v
    for layer_num, ca in enumerate(ca_layers, 1):
        if layers_to_edit is not None and layer_num-1 not in layers_to_edit:
            continue
        print(f"[EBF-Bures|ProjErase+RowWeight+UCE] Editing to_v   {layer_num}/{len(ca_layers)}")
        with torch.no_grad():
            W_old = ca.to_v.weight.data.clone()
            out_dim, in_dim = W_old.shape

            # (a) Row entropy -> e_i, H_row
            e_i = _compute_entropy_factor(W_old, concept_vecs,
                                          num_samples=entropy_samples,
                                          bins=entropy_bins,
                                          noise_sigma=noise_sigma)
            H_row = (1.0 - e_i).clamp(min=0.0, max=1.0)

            # (b) mu (global)
            mu = H_row.mean().item() if bures_mu_from_entropy else 0.1

            # (c) ASED spectral regularization
            R_ased = _ased_regularizer_from_C(C_full, T_sigma=T_sigma, p_sigma=p_sigma,
                                              strength=mu, eps=eps) if enable_ased else torch.zeros(in_dim, in_dim, device=device)

            # (d1) Aggregated MI row weights m_i for Pi_C (max over all concepts)
            if use_mi_softmask:
                row_ws_all = []
                for c_vec in concept_vecs:
                    row_ws_all.append(_compute_mi_softmask_emptyneg(
                        W_old     = W_old,
                        c_vec     = c_vec,
                        empty_vec = empty_vec,
                        num_pos   = 5,
                        T         = 0.7,
                        p         = p,
                        noise_sigma = noise_sigma
                    ))
                row_w_max = torch.max(torch.stack(row_ws_all, dim=-1), dim=-1).values   # (out_dim, 1)
            else:
                row_w_max = torch.ones((out_dim,1), device=W_old.device)

            # (d2) -- fx3-style UCE accumulation terms (for current layer) --
            # Accumulate mat1/mat2, then extract V and S, later merged into M_i and b_i
            I_in = torch.eye(in_dim, device=device)
            mat1_agg = lamb * W_old.clone()
            mat2_agg = lamb * I_in + CCt

            for idx_concept, (ot, nt, c_vec) in enumerate(zip(old_texts, new_texts, concept_vecs), 1):
                tinp = tok([ot, nt], padding="max_length",
                           max_length=tok.model_max_length,
                           truncation=True, return_tensors="pt")
                emb  = enc(tinp.input_ids.to(device))[0]
                f_o  = tinp.attention_mask[0].sum().item() - 2
                f_n  = tinp.attention_mask[1].sum().item() - 2
                far  = max(f_o, f_n)
                old_emb = emb[0][f_o: len(emb[0]) - max(0, far - f_o)]
                new_emb = emb[1][f_n: len(emb[1]) - max(0, far - f_n)]
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
                for_mat1 = (val_v @ ctx_vT).sum(0)      # (in_dim, in_dim)
                for_mat2 = (ctx_v  @ ctx_vT).sum(0)     # (in_dim, in_dim)

                if use_mi_softmask:
                    row_w_c = _compute_mi_softmask_emptyneg(
                        W_old     = W_old,
                        c_vec     = c_vec,
                        empty_vec = empty_vec,
                        num_pos   = 5,
                        T         = 0.7,
                        p         = p,
                        noise_sigma = noise_sigma
                    )
                else:
                    row_w_c = torch.ones((out_dim,1), device=W_old.device)

                # Note: accumulation method consistent with fx3
                mat1_agg += erase_scale * (for_mat1 * row_w_c)
                mat2_agg += erase_scale * for_mat2

            # Extract the incremental part of UCE accumulation
            V = mat1_agg - lamb * W_old            # (out_dim, in_dim)  row increment
            S = mat2_agg - (lamb * I_in + CCt)     # (in_dim, in_dim)   quadratic term increment

            # (e) Assemble hybrid per-row SPD system: M_i w = b_i
            base_SPD = (lamb + mu) * I_in + R_ased
            G_base   = base_SPD + CCt + S

            We = torch.empty_like(W_old)
            for i in range(out_dim):
                alpha_i = float(erase_scale * row_w_max[i].item())  # row-weighted subspace strength
                M_i = G_base + alpha_i * PiC                        # (in_dim, in_dim) SPD
                b_i = ((lamb + mu) * W_old[i] + V[i]).unsqueeze(1)  # (in_dim, 1)

                Lchol = torch.linalg.cholesky(M_i)
                we_i  = torch.cholesky_solve(b_i, Lchol).squeeze(1)  # (in_dim,)
                We[i] = we_i

            # (f) Bures proximal
            lam_prox = (lamb + mu)
            W_bures = bures_prox_row(W0=W_old, We=We,
                                     mu=mu, lam_prox=lam_prox,
                                     iters=bures_iters, eps=eps)

            # (g) Optional safety valve: EGBR
            if enable_egbr:
                W_final = egbr_row(W_old, W_bures, H_row)
            else:
                W_final = W_bures

            ca.to_v.weight.data.copy_(W_final)

    # === 4-B. Edit to_k (also fusing UCE accumulation + subspace projection-writeback)
    if with_to_k:
        for layer_num, ca in enumerate(ca_layers, 1):
            if layers_to_edit is not None and layer_num-1 not in layers_to_edit:
                continue
            print(f"[EBF-Bures|ProjErase+RowWeight+UCE] Editing to_k   {layer_num}/{len(ca_layers)}")
            with torch.no_grad():
                W_old = ca.to_k.weight.data.clone()
                out_dim, in_dim = W_old.shape

                e_i = _compute_entropy_factor(W_old, concept_vecs,
                                              num_samples=entropy_samples,
                                              bins=entropy_bins,
                                              noise_sigma=noise_sigma)
                H_row = (1.0 - e_i).clamp(min=0.0, max=1.0)
                mu = H_row.mean().item() if bures_mu_from_entropy else 0.1

                R_ased = _ased_regularizer_from_C(C_full, T_sigma=T_sigma, p_sigma=p_sigma,
                                                  strength=mu, eps=eps) if enable_ased else torch.zeros(in_dim, in_dim, device=device)

                # (k1) Aggregated row weights for Pi_C (max over concepts)
                if use_mi_softmask:
                    row_ws_all = []
                    for c_vec in concept_vecs:
                        row_ws_all.append(_compute_mi_softmask_emptyneg(
                            W_old     = W_old,
                            c_vec     = c_vec,
                            empty_vec = empty_vec,
                            num_pos   = 5,
                            T         = 0.7,
                            p         = p,
                            noise_sigma = noise_sigma
                        ))
                    row_w_max = torch.max(torch.stack(row_ws_all, dim=-1), dim=-1).values
                else:
                    row_w_max = torch.ones((out_dim,1), device=W_old.device)

                # (k2) UCE accumulation (to_k version)
                I_in = torch.eye(in_dim, device=device)
                mat1_agg = lamb * W_old.clone()
                mat2_agg = lamb * I_in + CCt

                for idx_concept, (ot, nt, c_vec) in enumerate(zip(old_texts, new_texts, concept_vecs), 1):
                    tinp = tok([ot, nt], padding="max_length",
                               max_length=tok.model_max_length,
                               truncation=True, return_tensors="pt")
                    emb  = enc(tinp.input_ids.to(device))[0]
                    f_o  = tinp.attention_mask[0].sum().item() - 2
                    f_n  = tinp.attention_mask[1].sum().item() - 2
                    far  = max(f_o, f_n)
                    old_emb = emb[0][f_o: len(emb[0]) - max(0, far - f_o)]
                    new_emb = emb[1][f_n: len(emb[1]) - max(0, far - f_n)]
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

                    if use_mi_softmask:
                        row_w_c = _compute_mi_softmask_emptyneg(
                            W_old     = W_old,
                            c_vec     = c_vec,
                            empty_vec = empty_vec,
                            num_pos   = 5,
                            T         = 0.7,
                            p         = p,
                            noise_sigma = noise_sigma
                        )
                    else:
                        row_w_c = torch.ones((out_dim,1), device=W_old.device)

                    mat1_agg += erase_scale * (for_mat1 * row_w_c)
                    mat2_agg += erase_scale * for_mat2

                V = mat1_agg - lamb * W_old
                S = mat2_agg - (lamb * I_in + CCt)

                base_SPD = (lamb + mu) * I_in + R_ased
                G_base   = base_SPD + CCt + S

                We = torch.empty_like(W_old)
                for i in range(out_dim):
                    alpha_i = float(erase_scale * row_w_max[i].item())
                    M_i = G_base + alpha_i * PiC
                    b_i = ((lamb + mu) * W_old[i] + V[i]).unsqueeze(1)

                    Lchol = torch.linalg.cholesky(M_i)
                    we_i  = torch.cholesky_solve(b_i, Lchol).squeeze(1)
                    We[i] = we_i

                lam_prox = (lamb + mu)
                W_bures = bures_prox_row(W0=W_old, We=We,
                                         mu=mu, lam_prox=lam_prox,
                                         iters=bures_iters, eps=eps)
                if enable_egbr:
                    W_final = egbr_row(W_old, W_bures, H_row)
                else:
                    W_final = W_bures
                ca.to_k.weight.data.copy_(W_final)

    print(f'[edit_model] done | enable_ased={enable_ased} | enable_egbr={enable_egbr} | use_mi_softmask={use_mi_softmask} | bures_iters={bures_iters}')
    return ldm_stable

# ─────────────────────────────────────────────────────────────
# Main entry point (unchanged)
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='TrainUSD',
                    description='Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concepts', type=str, required=True)
    parser.add_argument('--guided_concepts', type=str, default=None)
    parser.add_argument('--preserve_concepts', type=str, default=None)
    parser.add_argument('--technique', type=str, default='replace')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--base', type=str, default='1.4')
    parser.add_argument('--preserve_scale', type=float, default=None)
    parser.add_argument('--preserve_number', type=int, default=None)
    parser.add_argument('--erase_scale', type=float, default=1)
    parser.add_argument('--concept_type', type=str, required=True)
    parser.add_argument('--add_prompts', type=bool, default=False)

    parser.add_argument('--svd', action='store_true', default=False)
    parser.add_argument('--p', type=float, default=2.0)
    parser.add_argument('--res_scale', type=float, default=None)
    parser.add_argument('--alpha_min', type=float, default=0.1)

    parser.add_argument("--entropy_samples", type=int, default=20)
    parser.add_argument("--entropy_bins", type=int, default=20)
    parser.add_argument("--noise_sigma", type=float, default=0.01)

    parser.add_argument('--T_sigma', type=float, default=1)
    parser.add_argument('--p_sigma', type=float, default=1)

    parser.add_argument('--enable_ased', action='store_true', default=False)
    parser.add_argument('--enable_egbr', action='store_true', default=False)
    parser.add_argument('--bures_mu_from_entropy', action='store_true', default=True)
    parser.add_argument('--use_mi_softmask', action='store_true', default=False)

    # Bures proximal iteration count
    parser.add_argument('--bures_iters', type=int, default=2, help='Bures proximal iterations per row (2–3 is enough)')

    args = parser.parse_args()
    technique = args.technique
    device = f'cuda:{args.device}'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    add_prompts = args.add_prompts
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    preserve_number = args.preserve_number
    concepts = [con.strip() for con in args.concepts.split(',')]
    concept_type = args.concept_type

    print_text = ''
    for txt in concepts:
        print_text += txt.lower() + '_'
    print_text = print_text[:-1]

    if concepts[0] == 'allartist':
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = 5
        concepts = random.sample(artists, number)
    if concepts[0] == 'i2g':
        concepts = ['hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide', 'sexual', 'nudity', 'bodily fluids', 'blood']
    if concepts[0] == '10artists':
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = 10
        concepts = random.sample(artists, number)

    old_texts = []
    additional_prompts = []
    if concept_type == 'art':
        additional_prompts += ['painting by {concept}','art by {concept}','artwork by {concept}','picture by {concept}','style of {concept}']
    elif concept_type=='object':
        additional_prompts += ['image of {concept}','photo of {concept}','portrait of {concept}','picture of {concept}','painting of {concept}']
    if not add_prompts: additional_prompts = []
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
            new_texts = reduce(operator.concat, [[con]*len(old_texts) for con in guided_concepts])
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
    if len(retain_texts) > 1: print_text += f'-preserve_true'
    else: print_text += f'-preserve_false'
    if preserve_scale is None: preserve_scale = max(0.1, 1/len(retain_texts))

    sd14 = "CompVis/stable-diffusion-v1-4"
    sd15 = "runwayml/stable-diffusion-v1-5"
    sd21 = 'stabilityai/stable-diffusion-2-1-base'
    if args.base == '1.4': model_version = sd14
    elif args.base == '1.5': model_version = sd15
    elif args.base == '2.1': model_version = sd21
    else: model_version = sd14
    print(f"Model selection: using base version {args.base} ({model_version})")
    print("Loading Stable Diffusion pipeline...")
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    print("Stable Diffusion model loaded successfully.")
    print_text += f"-sd_{args.base.replace('.','_')}"
    print_text += f"-method_{technique}"
    print(print_text)
    print("Starting model editing...")

    ldm_stable = edit_model(
        ldm_stable=ldm_stable, old_text_=old_texts, new_text_=new_texts,
        with_to_k=True,
        add=False, retain_text_=retain_texts, lamb=0.5,
        erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique,
        svd=args.svd, p=args.p, res_scale=args.res_scale,
        alpha_min=args.alpha_min,
        entropy_samples=args.entropy_samples,
        entropy_bins=args.entropy_bins,
        noise_sigma=args.noise_sigma,
        T_sigma=args.T_sigma,
        p_sigma=args.p_sigma,
        enable_ased=args.enable_ased,
        enable_egbr=args.enable_egbr,
        bures_mu_from_entropy=args.bures_mu_from_entropy,
        use_mi_softmask=args.use_mi_softmask,
        bures_iters=args.bures_iters
    )

    print("Model editing completed.")
    torch.save(ldm_stable.unet.state_dict(), f'models/your_saved_model.pt')