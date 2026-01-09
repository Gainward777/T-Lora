import inspect
import math
import weakref
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file as safetensors_save_file
    from safetensors.torch import load_file as safetensors_load_file
except Exception:  # pragma: no cover
    safetensors_save_file = None
    safetensors_load_file = None


@dataclass
class TLoRAConfig:
    rank: int
    alpha: float = 1.0
    dropout: float = 0.0
    min_rank: int = 1
    alpha_rank_scale: float = 1.0
    trainer_type: str = "lora"  # "lora" | "ortho_lora"
    sig_type: str = "principal"  # "principal" | "last" | "middle"


def sigma_mask_from_timestep(
    timestep: int,
    *,
    max_timestep: int,
    rank: int,
    min_rank: int = 1,
    alpha_rank_scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Mirror authors' SDXL implementation:
      r = int(((T - t)/T)**alpha * (rank - min_rank)) + min_rank
      mask[:r] = 1 else 0
    """
    rank = int(rank)
    min_rank = int(min_rank)
    max_timestep = int(max_timestep)
    t = int(timestep)
    if rank <= 0:
        raise ValueError("rank must be > 0")
    if max_timestep <= 0:
        r = rank
    else:
        min_rank = max(1, min(min_rank, rank))
        frac = ((max_timestep - t) / max_timestep) ** float(alpha_rank_scale)
        r = int(frac * (rank - min_rank)) + min_rank
        r = max(1, min(r, rank))
    mask = torch.zeros((1, rank), device=device, dtype=dtype or torch.float32)
    mask[:, :r] = 1.0
    return mask


def sigma_mask_from_step_index(
    step_idx: int,
    *,
    num_inference_steps: int,
    rank: int,
    min_rank: int = 1,
    alpha_rank_scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Scheduler-agnostic: uses progress through denoising steps.
    step_idx=0 -> most noisy -> smallest rank.
    """
    n = int(num_inference_steps)
    if n <= 1:
        r = int(rank)
    else:
        p = float(step_idx) / float(n - 1)  # 0..1
        frac = (1.0 - p) ** float(alpha_rank_scale)
        min_rank = max(1, min(int(min_rank), int(rank)))
        r = int(frac * (int(rank) - min_rank)) + min_rank
        r = max(1, min(r, int(rank)))
    mask = torch.zeros((1, int(rank)), device=device, dtype=dtype or torch.float32)
    mask[:, :r] = 1.0
    return mask


class TLoRALinear(nn.Module):
    """
    Base linear + LoRA with rank-masking controlled via transformer attribute `_tlora_sigma_mask`.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        in_features = base.in_features
        out_features = base.out_features
        rank = int(rank)
        if rank > min(in_features, out_features):
            raise ValueError(f"rank {rank} must be <= min(in,out)={min(in_features, out_features)}")

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else None

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        # LoRA init: down ~ N(0, 1/r), up = 0 => start from base model exactly
        nn.init.normal_(self.down.weight, std=1.0 / max(1.0, float(rank)))
        nn.init.zeros_(self.up.weight)

        # Freeze base weights
        self.base.requires_grad_(False)

        # Populated by injector
        # IMPORTANT: don't store owner as nn.Module attribute; otherwise PyTorch registers it as a submodule
        # and creates a cyclic module graph (transformer -> tlora -> transformer), causing RecursionError in .train().
        self.__dict__["_tlora_owner_ref"] = None  # type: ignore[assignment]

    def set_owner(self, owner: nn.Module):
        # Store a weakref in __dict__ to bypass nn.Module.__setattr__ registration.
        self.__dict__["_tlora_owner_ref"] = weakref.ref(owner)

    def _get_sigma_mask(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        owner_ref = self.__dict__.get("_tlora_owner_ref")
        owner = owner_ref() if callable(owner_ref) else None
        if owner is None:
            return None
        m = getattr(owner, "_tlora_sigma_mask", None)
        if m is None:
            return None
        if not torch.is_tensor(m):
            return None
        # m expected shape (1, Rmax). Slice for this layer.
        m = m.to(device=device, dtype=dtype)
        if m.shape[-1] != self.rank:
            m = m[..., : self.rank]
        return m

    def _get_global_scale(self) -> float:
        owner_ref = self.__dict__.get("_tlora_owner_ref")
        owner = owner_ref() if callable(owner_ref) else None
        if owner is None:
            return 1.0
        try:
            return float(getattr(owner, "_tlora_scale", 1.0))
        except Exception:
            return 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        sigma_mask = self._get_sigma_mask(device=x.device, dtype=self.down.weight.dtype)
        if sigma_mask is None:
            sigma_mask = torch.ones((1, self.rank), device=x.device, dtype=self.down.weight.dtype)
        h = x
        if self.dropout is not None:
            h = self.dropout(h)
        down = self.down(h.to(self.down.weight.dtype)) * sigma_mask
        up = self.up(down)
        return out + (up.to(out.dtype) * self.scaling * self._get_global_scale())


class OrthoTLoRALinear(nn.Module):
    """
    Ortho-LoRA adapted from authors' implementation:
    - init using SVD of base weight
    - return delta that starts at 0 via base subtraction
    Masking applied the same way as Vanilla (sigma_mask on rank components).
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float = 1.0, sig_type: str = "principal"):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")

        in_features = base.in_features
        out_features = base.out_features
        rank = int(rank)
        if rank > min(in_features, out_features):
            raise ValueError(f"rank {rank} must be <= min(in,out)={min(in_features, out_features)}")

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.sig_type = str(sig_type)

        self.q_layer = nn.Linear(in_features, rank, bias=False)  # rank x in
        self.p_layer = nn.Linear(rank, out_features, bias=False)  # out x rank
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))

        # Same cycle-avoidance as in TLoRALinear
        self.__dict__["_tlora_owner_ref"] = None  # type: ignore[assignment]

        # Freeze base weights
        self.base.requires_grad_(False)

        # Initialize from SVD(base.weight) to encourage orthogonality / full effective rank
        # NOTE: torch.linalg.svd returns (U, S, Vh)
        with torch.no_grad():
            w = self.base.weight.detach().float()  # (out, in)
            u, s, vh = torch.linalg.svd(w, full_matrices=True)

            def _take(sig_type_: str):
                if sig_type_ == "principal":
                    u_sel = u[:, :rank]
                    vh_sel = vh[:rank, :]
                    s_sel = s[:rank]
                elif sig_type_ == "last":
                    u_sel = u[:, -rank:]
                    vh_sel = vh[-rank:, :]
                    s_sel = s[-rank:]
                elif sig_type_ == "middle":
                    su = u.shape[1]
                    sv = vh.shape[0]
                    ss = s.shape[0]
                    u_start = math.ceil((su - rank) / 2)
                    v_start = math.ceil((sv - rank) / 2)
                    s_start = math.ceil((ss - rank) / 2)
                    u_sel = u[:, u_start : u_start + rank]
                    vh_sel = vh[v_start : v_start + rank, :]
                    s_sel = s[s_start : s_start + rank]
                else:
                    raise ValueError("sig_type must be one of: principal|last|middle")
                return u_sel, s_sel, vh_sel

            u_sel, s_sel, vh_sel = _take(self.sig_type)
            self.q_layer.weight.copy_(vh_sel)  # (rank, in)
            self.p_layer.weight.copy_(u_sel)  # (out, rank)
            self.lambda_layer.copy_(s_sel.unsqueeze(0))  # (1, rank)

        # Base copies (frozen) to ensure delta starts at 0
        self.register_buffer("base_q_weight", self.q_layer.weight.detach().clone())
        self.register_buffer("base_p_weight", self.p_layer.weight.detach().clone())
        self.register_buffer("base_lambda", self.lambda_layer.detach().clone())

    def set_owner(self, owner: nn.Module):
        self.__dict__["_tlora_owner_ref"] = weakref.ref(owner)

    def _get_sigma_mask(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        owner_ref = self.__dict__.get("_tlora_owner_ref")
        owner = owner_ref() if callable(owner_ref) else None
        if owner is None:
            return None
        m = getattr(owner, "_tlora_sigma_mask", None)
        if m is None or (not torch.is_tensor(m)):
            return None
        m = m.to(device=device, dtype=dtype)
        if m.shape[-1] != self.rank:
            m = m[..., : self.rank]
        return m

    def _get_global_scale(self) -> float:
        owner_ref = self.__dict__.get("_tlora_owner_ref")
        owner = owner_ref() if callable(owner_ref) else None
        if owner is None:
            return 1.0
        try:
            return float(getattr(owner, "_tlora_scale", 1.0))
        except Exception:
            return 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dtype = self.q_layer.weight.dtype
        sigma_mask = self._get_sigma_mask(device=x.device, dtype=dtype)
        if sigma_mask is None:
            sigma_mask = torch.ones((1, self.rank), device=x.device, dtype=dtype)

        # Trainable branch
        q = self.q_layer(x.to(dtype)) * self.lambda_layer * sigma_mask
        p = self.p_layer(q)

        # Frozen base branch (same mask!)
        base_q = torch.matmul(x.to(dtype), self.base_q_weight.t()) * self.base_lambda * sigma_mask
        base_p = torch.matmul(base_q, self.base_p_weight.t())

        delta = p - base_p
        return out + (delta.to(out.dtype) * self.scaling * self._get_global_scale())


def _iter_named_linears(root: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, m in root.named_modules():
        if isinstance(m, nn.Linear):
            yield name, m


def inject_tlora_into_transformer(
    transformer: nn.Module,
    *,
    target_module_suffixes: List[str],
    cfg: TLoRAConfig,
) -> int:
    """
    Replaces Linear layers (to_q/to_k/to_v/...) with TLoRA wrappers.
    Returns number of injected modules.
    """
    suffixes = list(target_module_suffixes)
    if not suffixes:
        raise ValueError("target_module_suffixes cannot be empty")

    # Collect replacements first to avoid mutating while iterating.
    replacements: List[Tuple[nn.Module, str, nn.Linear]] = []
    for full_name, lin in _iter_named_linears(transformer):
        last = full_name.split(".")[-1]
        if last in suffixes or any(full_name.endswith(suf) for suf in suffixes):
            # Find parent module and attribute name
            parts = full_name.split(".")
            parent = transformer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            replacements.append((parent, attr, lin))

    injected = 0
    for parent, attr, lin in replacements:
        if cfg.trainer_type == "ortho_lora":
            wrapped: nn.Module = OrthoTLoRALinear(
                lin, rank=cfg.rank, alpha=cfg.alpha, sig_type=cfg.sig_type
            )
        else:
            wrapped = TLoRALinear(lin, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
        # Owner is transformer; wrappers will read transformer._tlora_sigma_mask and transformer._tlora_scale
        if hasattr(wrapped, "set_owner"):
            wrapped.set_owner(transformer)  # type: ignore[attr-defined]
        setattr(parent, attr, wrapped)
        injected += 1

    if injected == 0:
        raise RuntimeError(f"No Linear layers matched suffixes={suffixes}. Check target_modules.")

    # Default scale
    if not hasattr(transformer, "_tlora_scale"):
        setattr(transformer, "_tlora_scale", 1.0)
    return injected


def patch_transformer_forward_for_joint_attention_kwargs(transformer: nn.Module) -> None:
    """
    Monkey-patch transformer.forward to capture joint_attention_kwargs['sigma_mask'] (if present)
    into transformer._tlora_sigma_mask, without requiring any diffusers fork.

    Also captures joint_attention_kwargs['scale'] into transformer._tlora_scale (optional).
    """
    if getattr(transformer, "_tlora_forward_patched", False):
        return

    orig_forward = transformer.forward
    sig = inspect.signature(orig_forward)
    accepted = set(sig.parameters.keys())

    def _filtered_kwargs(kwargs: Dict) -> Dict:
        # Keep only kwargs that original forward accepts; avoids breaking if we add extra keys.
        return {k: v for k, v in kwargs.items() if k in accepted}

    def wrapped_forward(*args, **kwargs):
        ja = kwargs.get("joint_attention_kwargs", None)
        if isinstance(ja, dict):
            if "sigma_mask" in ja:
                transformer._tlora_sigma_mask = ja["sigma_mask"]
            if "scale" in ja:
                transformer._tlora_scale = float(ja["scale"])
        # Some versions might use cross_attention_kwargs; accept it too just in case.
        ca = kwargs.get("cross_attention_kwargs", None)
        if isinstance(ca, dict):
            if "sigma_mask" in ca:
                transformer._tlora_sigma_mask = ca["sigma_mask"]
            if "scale" in ca:
                transformer._tlora_scale = float(ca["scale"])
        return orig_forward(*args, **_filtered_kwargs(kwargs))

    transformer.forward = wrapped_forward  # type: ignore[assignment]
    transformer._tlora_forward_patched = True


def tlora_parameters(module: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, (TLoRALinear, OrthoTLoRALinear)):
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
    return params


def tlora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    sd = module.state_dict()
    keep: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        # Heuristic: keep only keys belonging to our wrappers and their buffers.
        if ".down." in k or ".up." in k or ".q_layer." in k or ".p_layer." in k or "lambda_layer" in k:
            if "base_q_weight" in k or "base_p_weight" in k or "base_lambda" in k:
                # Keep base buffers too for deterministic inference
                keep[k] = v
            else:
                keep[k] = v
    return keep


def save_tlora_weights(module: nn.Module, out_file: str) -> None:
    sd = tlora_state_dict(module)
    if safetensors_save_file is None:
        torch.save(sd, out_file)
        return
    safetensors_save_file(sd, out_file)


def load_tlora_weights(module: nn.Module, in_file: str, strict: bool = False) -> None:
    if safetensors_load_file is None:
        sd = torch.load(in_file, map_location="cpu")
    else:
        sd = safetensors_load_file(in_file)
    missing, unexpected = module.load_state_dict(sd, strict=strict)
    # Best-effort load: allow missing/unexpected due to base model differences
    if strict and (missing or unexpected):
        raise RuntimeError(f"Failed to load T-LoRA weights. missing={missing}, unexpected={unexpected}")


