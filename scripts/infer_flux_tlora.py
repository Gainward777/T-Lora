import os
import argparse
import yaml
import torch

from diffusers import FluxPipeline

from tlora_flux import (
    TLoRAConfig,
    inject_tlora_into_transformer,
    patch_transformer_forward_for_joint_attention_kwargs,
    sigma_mask_from_step_index,
    load_tlora_weights,
)


    
NEUTRAL_TRIGGER_PROMPTS = [
    "a modern bathroom interior, realistic interior render, soft natural lighting, minimalist design, "
    "matbrk white subway brick wall",

    "a modern living room interior, realistic interior render, clean architectural style, warm ambient lighting, "
    "matbrk white subway brick wall",

    "a minimalist bedroom interior, realistic architectural render, soft daylight, calm neutral tones, "
    "matbrk white subway brick accent wall",

    "interior detail, wall surface close-up, photorealistic material study, realistic lighting, high detail, "
    "matbrk white subway brick wall",

    "wall surface material study, photorealistic, realistic lighting, high detail, "
    "matbrk white subway brick wall",
]




def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--lora_path", type=str, required=True)  # outputs/.../final or checkpoint-300
    ap.add_argument("--outdir", type=str, default="outputs/infer")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num", type=int, default=2)
    ap.add_argument("--hf_token", type=str, default=None)

    # A/B test (recommended)
    ap.add_argument("--ab_test", action="store_true",
                    help="Generate variants per prompt: scales from --ab_scales")
    ap.add_argument("--ab_scales", type=str, default="0.0,0.7,1.0",
                    help="Comma-separated scales for --ab_test (default: 0.0,0.7,1.0)")

    # Single-scale mode
    ap.add_argument("--lora_scale", type=float, default=1.0,
                    help="used when --ab_test is NOT set")

    ap.add_argument("--sequential_offload", action="store_true",
                    help="even lower VRAM (slower) than model_cpu_offload")
    return ap.parse_args()


def _set_tlora_scale(transformer, scale: float):
    # Read by patched transformer.forward and/or wrappers (best-effort).
    setattr(transformer, "_tlora_scale", float(scale))


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    steps = args.steps if args.steps is not None else int(cfg.get("num_inference_steps", 25))
    guidance = args.guidance if args.guidance is not None else float(cfg.get("guidance_scale", 4.0))

    mp = (cfg.get("mixed_precision", "bf16") or "").lower()
    dtype = torch.bfloat16 if mp in ("bf16", "bfloat16") else torch.float16

    # IMPORTANT: do NOT pipe.to("cuda") for FLUX.1-dev
    pipe = FluxPipeline.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        use_safetensors=True,
        token=args.hf_token,
    )

    # reduce VAE peaks
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    # Inject T-LoRA layers into transformer and load weights
    tlora_cfg = TLoRAConfig(
        rank=int(cfg.get("lora_rank", 16)),
        alpha=float(cfg.get("lora_alpha", cfg.get("lora_rank", 16))),
        dropout=float(cfg.get("lora_dropout", 0.0)),
        min_rank=int(cfg.get("min_rank", max(1, int(cfg.get("lora_rank", 16)) // 2))),
        alpha_rank_scale=float(cfg.get("alpha_rank_scale", 1.0)),
        trainer_type=str(cfg.get("trainer_type", "lora")),
        sig_type=str(cfg.get("sig_type", "principal")),
    )
    target_modules = cfg.get("target_modules", ["to_q", "to_k", "to_v"])
    inject_tlora_into_transformer(
        pipe.transformer, target_module_suffixes=list(target_modules), cfg=tlora_cfg
    )
    patch_transformer_forward_for_joint_attention_kwargs(pipe.transformer)

    weights_path = os.path.join(args.lora_path, "tlora_weights.safetensors")
    if not os.path.exists(weights_path):
        # allow pointing directly to file
        if args.lora_path.endswith(".safetensors") and os.path.exists(args.lora_path):
            weights_path = args.lora_path
        else:
            raise FileNotFoundError(f"Could not find tlora weights at: {weights_path}")
    load_tlora_weights(pipe.transformer, weights_path, strict=False)

    # Enable offload after PEFT wrapping (prevents OOM)
    if args.sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    prompts = NEUTRAL_TRIGGER_PROMPTS

    os.makedirs(args.outdir, exist_ok=True)
    gen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ab_test:
        scales = [float(s.strip()) for s in args.ab_scales.split(",") if s.strip() != ""]
        if not scales:
            scales = [0.0, 0.7, 1.0]
    else:
        scales = [float(args.lora_scale)]

    for p_i, prompt in enumerate(prompts):
        for scale in scales:
            _set_tlora_scale(pipe.transformer, scale)

            # same seed per scale -> fair A/B comparison
            g = torch.Generator(device=gen_device).manual_seed(args.seed)

            for i in range(args.num):
                # Mutable dict so callback can update it in-place each step.
                joint_attention_kwargs = {"sigma_mask": None, "scale": float(scale)}

                def _cb_on_step_end(pipeline, step_idx, timestep, callback_kwargs):
                    # Compute sigma_mask based on step index (scheduler-agnostic).
                    sigma_mask = sigma_mask_from_step_index(
                        int(step_idx),
                        num_inference_steps=int(steps),
                        rank=int(tlora_cfg.rank),
                        min_rank=int(tlora_cfg.min_rank),
                        alpha_rank_scale=float(tlora_cfg.alpha_rank_scale),
                        device=gen_device,
                        dtype=torch.float32,
                    )
                    joint_attention_kwargs["sigma_mask"] = sigma_mask
                    # Also set attribute for wrappers (best-effort, in case kwargs path changes)
                    pipe.transformer._tlora_sigma_mask = sigma_mask
                    return callback_kwargs

                image = pipe(
                    prompt=prompt,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=g,
                    joint_attention_kwargs=joint_attention_kwargs,
                    callback_on_step_end=_cb_on_step_end,
                ).images[0]

                mode = "ab" if args.ab_test else "single"
                fn = f"{mode}_neutraltrigger_p{p_i:02d}_img{i:02d}_steps{steps}_g{guidance}_lora{scale:.2f}.png"
                path = os.path.join(args.outdir, fn)
                image.save(path)
                print("saved:", path)


if __name__ == "__main__":
    main()
