import os
import random
import argparse
import yaml
from typing import List, Tuple, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 (оставлено, не мешает)
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from accelerate import Accelerator
from tqdm import tqdm

from diffusers import FluxPipeline

# ---------------------------
# PyTorch SDP compat (diffusers may pass enable_gqa on newer torch)
# ---------------------------
#
# Some diffusers versions call torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=...).
# On some torch builds this kwarg doesn't exist and crashes at runtime.
try:
    if not getattr(torch.nn.functional, "_tlora_sdp_enable_gqa_patched", False):
        _orig_sdp = torch.nn.functional.scaled_dot_product_attention

        def _sdp_compat(*args, **kwargs):
            # Ignore 'enable_gqa' if the current torch build doesn't support it.
            # Safe even if torch supports it: we just fall back to default behavior.
            kwargs.pop("enable_gqa", None)
            return _orig_sdp(*args, **kwargs)

        torch.nn.functional.scaled_dot_product_attention = _sdp_compat  # type: ignore[assignment]
        torch.nn.functional._tlora_sdp_enable_gqa_patched = True  # type: ignore[attr-defined]
except Exception:
    # Best-effort only; if patching fails, training will surface the underlying error.
    pass

from tlora_flux import (
    TLoRAConfig,
    inject_tlora_into_transformer,
    patch_transformer_forward_for_joint_attention_kwargs,
    sigma_mask_from_timestep,
    tlora_parameters,
    save_tlora_weights,
)


# ---------------------------
# Seed
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Image helpers
# ---------------------------

def _resize_and_crop(img: Image.Image, resolution: int, center_crop: bool):
    w, h = img.size
    scale = resolution / min(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BICUBIC)

    if center_crop:
        left = (nw - resolution) // 2
        top = (nh - resolution) // 2
    else:
        left = random.randint(0, max(0, nw - resolution))
        top = random.randint(0, max(0, nh - resolution))

    return img.crop((left, top, left + resolution, top + resolution))


def load_image(path: str, resolution: int, center_crop: bool, random_flip: bool) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = _resize_and_crop(img, resolution, center_crop)
    if random_flip and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW


# ---------------------------
# Dataset (NO MASKS)
# ---------------------------

class CaptionDataset(Dataset):
    def __init__(self, data_dir: str, resolution: int, center_crop: bool, random_flip: bool):
        self.data_dir = data_dir
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.images = []
        for fn in sorted(os.listdir(data_dir)):
            low = fn.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
                self.images.append(fn)
        if not self.images:
            raise RuntimeError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        base, _ = os.path.splitext(img_name)

        img_path = os.path.join(self.data_dir, img_name)
        txt_path = os.path.join(self.data_dir, base + ".txt")

        image = load_image(img_path, self.resolution, self.center_crop, self.random_flip)
        caption = open(txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(txt_path) else ""

        # IMPORTANT: no "mask" key at all (DataLoader can't collate None)
        return {"pixel_values": image, "caption": caption}


# ---------------------------
# Misc helpers
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=None)
    return ap.parse_args()


def timestep_scale(t: torch.Tensor, schedule: List[Tuple[int, int, float]]) -> torch.Tensor:
    out = torch.ones_like(t, dtype=torch.float32)
    for tmin, tmax, s in schedule:
        m = (t >= tmin) & (t <= tmax)
        out[m] = float(s)
    return out


def discover_linear_targets(model: nn.Module, requested: List[str]) -> List[str]:
    all_linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    safe = []
    for key in requested:
        matches = [n for n in all_linear_names if n.endswith(key) or n.split(".")[-1] == key or f".{key}." in n]
        if matches:
            safe.append(key)
    return safe or requested


@torch.no_grad()
def precompute_prompt_cache_cpu_then_to_gpu(
    pipe: FluxPipeline,
    captions: List[str],
    gpu_device: torch.device,
    dtype: torch.dtype,
    *,
    max_sequence_length: int = 77,
) -> Dict[str, tuple]:
    """
    FLUX in diffusers 0.36.0 expects txt_ids not None.
    Cache (prompt_embeds, pooled_prompt_embeds, txt_ids) on GPU.
    """
    cache: Dict[str, tuple] = {}
    for cap in sorted(set(captions)):
        pe_cpu, ppe_cpu, txt_ids_cpu = pipe.encode_prompt(
            [cap],
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            max_sequence_length=int(max_sequence_length),
        )
        cache[cap] = (
            pe_cpu.to(gpu_device, dtype=dtype, non_blocking=True),
            ppe_cpu.to(gpu_device, dtype=dtype, non_blocking=True),
            txt_ids_cpu.to(gpu_device, non_blocking=True),
        )
    return cache


def get_pack_latents_fn(pipe: FluxPipeline) -> Callable:
    for name in ["_pack_latents", "pack_latents"]:
        fn = getattr(pipe, name, None)
        if callable(fn):
            return fn
    raise RuntimeError("FluxPipeline has no _pack_latents/pack_latents. Update diffusers.")


def pack_latents_compat(pack_fn: Callable, latents: torch.Tensor) -> torch.Tensor:
    try:
        return pack_fn(latents)
    except TypeError:
        b, c, h, w = latents.shape
        return pack_fn(latents, b, c, h, w)


def make_noisy_latents_flowmatch(scheduler, latents, noise, timesteps):
    """
    FlowMatchEulerDiscreteScheduler has no add_noise().
    Use sigma mixing.
    """
    sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    sigma = sigmas[timesteps].view(-1, 1, 1, 1)
    noisy = latents * (1.0 - sigma) + noise * sigma

    if hasattr(scheduler, "scale_model_input"):
        try:
            noisy = scheduler.scale_model_input(noisy, timesteps)
        except Exception:
            pass
    return noisy


def _factor_hw(seq_len: int):
    h = int(np.sqrt(seq_len))
    while h > 1:
        if seq_len % h == 0:
            return h, seq_len // h
        h -= 1
    return 1, seq_len


def make_img_ids_for_seq(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Must be 2D: (seq_len, 3). seq_len must equal hidden_states.shape[1].
    """
    h, w = _factor_hw(seq_len)
    yy = torch.arange(h, device=device, dtype=torch.int64)
    xx = torch.arange(w, device=device, dtype=torch.int64)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    dummy = torch.zeros((h, w), device=device, dtype=torch.int64)
    return torch.stack([dummy, grid_y, grid_x], dim=-1).view(h * w, 3)


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
    )
    set_seed(int(cfg.get("seed", 0)))

    mp = (cfg.get("mixed_precision", "bf16") or "").lower()
    if mp in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif mp in ("fp16", "float16"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = accelerator.device

    pipe = FluxPipeline.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        use_safetensors=True,
        token=args.hf_token,
    )

    # Text encoders only for caching -> keep on CPU
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")

    # Trainable parts on GPU
    pipe.vae.to(device)
    pipe.transformer.to(device)

    # Memory reducers (without CPU offload)
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        try:
            pipe.transformer.enable_gradient_checkpointing()
        except Exception:
            pass

    if hasattr(pipe.vae, "enable_slicing"):
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    scheduler = pipe.scheduler
    pack_fn = get_pack_latents_fn(pipe)

    base_transformer = pipe.transformer
    base_transformer.requires_grad_(False)

    requested_targets = cfg.get("target_modules", ["to_q", "to_k", "to_v"])
    safe_targets = discover_linear_targets(base_transformer, requested_targets)

    if accelerator.is_local_main_process:
        print("Requested target_modules:", requested_targets)
        print("Safe target_modules (Linear-only):", safe_targets)

    # --- T-LoRA injection into FLUX transformer (QKV by default) ---
    tlora_cfg = TLoRAConfig(
        rank=int(cfg["lora_rank"]),
        alpha=float(cfg.get("lora_alpha", cfg["lora_rank"])),
        dropout=float(cfg.get("lora_dropout", 0.0)),
        min_rank=int(cfg.get("min_rank", max(1, int(cfg["lora_rank"]) // 2))),
        alpha_rank_scale=float(cfg.get("alpha_rank_scale", 1.0)),
        trainer_type=str(cfg.get("trainer_type", "lora")),
        sig_type=str(cfg.get("sig_type", "principal")),
    )

    injected = inject_tlora_into_transformer(
        base_transformer, target_module_suffixes=safe_targets, cfg=tlora_cfg
    )
    # Capture joint_attention_kwargs['sigma_mask'] in inference (and optionally scale)
    patch_transformer_forward_for_joint_attention_kwargs(base_transformer)

    transformer = base_transformer
    transformer.train()

    if accelerator.is_local_main_process:
        print(f"Injected T-LoRA modules: {injected}")

    ds = CaptionDataset(
        data_dir=cfg["train_data_dir"],
        resolution=int(cfg["resolution"]),
        center_crop=bool(cfg.get("center_crop", True)),
        random_flip=bool(cfg.get("random_flip", True)),
    )

    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Cache prompt embeds once (huge speedup for tiny dataset)
    all_caps = [ds[i]["caption"] for i in range(len(ds))]
    prompt_cache = precompute_prompt_cache_cpu_then_to_gpu(
        pipe,
        all_caps,
        gpu_device=device,
        dtype=dtype,
        max_sequence_length=int(cfg.get("max_sequence_length", 77)),
    )

    params = tlora_parameters(transformer)
    if not params:
        raise RuntimeError("No trainable T-LoRA parameters found after injection.")
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 5e-5)),
        betas=(float(cfg.get("adam_beta1", 0.9)), float(cfg.get("adam_beta2", 0.999))),
        weight_decay=float(cfg.get("adam_weight_decay", 0.0)),
        eps=float(cfg.get("adam_epsilon", 1e-8)),
    )

    max_steps = int(cfg.get("max_train_steps", 400))
    ckpt_every = int(cfg.get("checkpointing_steps", 100))
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    transformer, opt, dl = accelerator.prepare(transformer, opt, dl)

    num_ts = getattr(scheduler.config, "num_train_timesteps", 1000)
    enable_rank_masking = bool(cfg.get("enable_rank_masking", True))

    # Cache img_ids by (seq_len, device)
    img_ids_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    global_step = 0
    pbar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    pbar.set_description("train")

    while global_step < max_steps:
        for batch in dl:
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(device, dtype=dtype, non_blocking=True)
                captions = batch["caption"]

                prompt_embeds, pooled_prompt_embeds, txt_ids = prompt_cache[captions[0]]

                # Encode -> latents
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                bsz = latents.shape[0]
                timesteps = torch.randint(0, num_ts, (bsz,), device=device, dtype=torch.int64)

                noise = torch.randn_like(latents)
                noisy_latents = make_noisy_latents_flowmatch(scheduler, latents, noise, timesteps)

                # Pack for FLUX transformer
                hidden_states = pack_latents_compat(pack_fn, noisy_latents)
                seq_len = int(hidden_states.shape[1])

                k = (seq_len, str(device))
                img_ids = img_ids_cache.get(k)
                if img_ids is None:
                    img_ids = make_img_ids_for_seq(seq_len, device=device)
                    img_ids_cache[k] = img_ids

                # FLUX guidance-distilled expects guidance
                guidance = torch.zeros((hidden_states.shape[0],), device=device, dtype=pooled_prompt_embeds.dtype)

                # Flux forward expects timestep in [0..1]
                timestep_in = timesteps.to(hidden_states.dtype) / 1000.0

                if enable_rank_masking:
                    # Use integer timestep for mask computation (closest to authors' implementation).
                    # Note: timesteps are sampled uniformly in [0..num_ts)
                    t_int = int(timesteps[0].detach().cpu().item())
                    sigma_mask = sigma_mask_from_timestep(
                        t_int,
                        max_timestep=max(1, int(num_ts) - 1),
                        rank=int(tlora_cfg.rank),
                        min_rank=int(tlora_cfg.min_rank),
                        alpha_rank_scale=float(tlora_cfg.alpha_rank_scale),
                        device=device,
                        dtype=hidden_states.dtype,
                    )
                    # Store on transformer; wrappers will read it during forward.
                    pipe.transformer._tlora_sigma_mask = sigma_mask.detach()

                model_pred = pipe.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep_in,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                target = pack_latents_compat(pack_fn, noise)
                loss_map = (model_pred - target) ** 2
                loss = loss_map.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)

                opt.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.detach().cpu())})

                if accelerator.is_local_main_process and (global_step % ckpt_every == 0):
                    ckpt_path = os.path.join(out_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    save_tlora_weights(
                        accelerator.unwrap_model(transformer),
                        os.path.join(ckpt_path, "tlora_weights.safetensors"),
                    )

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    pbar.close()

    if accelerator.is_local_main_process:
        final_path = os.path.join(out_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        save_tlora_weights(
            accelerator.unwrap_model(transformer),
            os.path.join(final_path, "tlora_weights.safetensors"),
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
