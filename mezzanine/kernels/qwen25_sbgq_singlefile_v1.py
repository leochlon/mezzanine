from __future__ import annotations

"""
Calibration-free exact SBGQ for Qwen/Qwen2.5 text-only models.

This is the Qwen2.5-specific counterpart to the main Qwen3.5 script.
The important architectural simplifications are:

- Every decoder layer is text-only (no vision path / no language_model_only hacks).
- Attention is standard Qwen2 attention:
    q_proj, k_proj, v_proj, o_proj
- SwiGLU MLP is standard:
    gate_proj, up_proj, down_proj
- GQA is still present, so the exact attention gauge uses a KV-head permutation
  lifted to query-head groups, plus an exact value/output scaling symmetry.

Exact attention symmetry used here
----------------------------------
Let H_q be the number of query heads, H_kv the number of KV heads, g = H_q / H_kv,
and d_h the head dimension.

Choose:
- a permutation pi of KV heads,
- its lifted permutation on the query heads, which permutes groups of size g,
- a positive per-KV-head scale d.

Then the following is exact for Qwen2 attention:
- q_proj: permute query-head blocks
- k_proj: permute KV-head blocks
- v_proj: permute KV-head blocks and scale them by d
- o_proj: permute input query-head blocks and apply inverse lifted scale

This preserves the full-precision function because repeat_kv commutes with the
lifted permutation and the value scaling is exactly canceled in o_proj.

Gauge selection is calibration-free:
- the MLP gauge is computed from weights + RMSNorm scales only;
- the attention gauge is computed from v_proj / o_proj and input RMSNorm scales only.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
from typing import Iterator, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GaugeConfig:
    block_size: int = 64
    lambda_hidden: float = 1.0
    lambda_down: float = 1.0
    stat: Literal["mean_abs", "rms", "max_abs"] = "rms"
    hermite_order: int = 32
    scale_clip: float = 8.0


@dataclass(frozen=True)
class AttentionGaugeConfig:
    lambda_value: float = 1.0
    lambda_out: float = 1.0
    stat: Literal["mean_abs", "rms", "max_abs"] = "rms"
    scale_clip: float = 8.0
    permute_heads: bool = True
    scale_values: bool = True


@dataclass
class SwiGLUUpGauge:
    perm: torch.Tensor
    inv_perm: torch.Tensor
    block_scale: torch.Tensor
    block_size: int

    def expanded_scale(self, H: Optional[int] = None, *, device=None, dtype=torch.float32) -> torch.Tensor:
        if H is None:
            H = int(self.perm.numel())
        s = self.block_scale.to(device=device, dtype=torch.float32).repeat_interleave(self.block_size)
        s = s[:H]
        return s.to(dtype=dtype)


@dataclass
class AttentionHeadGauge:
    kv_head_perm: torch.Tensor
    kv_head_inv_perm: torch.Tensor
    kv_head_scale: torch.Tensor
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    group_size: int

    def query_head_perm(self) -> torch.Tensor:
        g = self.group_size
        perm = self.kv_head_perm.to(torch.int64)
        out = []
        for p in perm.tolist():
            base = p * g
            out.extend(range(base, base + g))
        return torch.tensor(out, dtype=torch.int64, device=perm.device)

    def query_head_inv_perm(self) -> torch.Tensor:
        qperm = self.query_head_perm()
        inv = torch.empty_like(qperm)
        inv[qperm] = torch.arange(qperm.numel(), device=qperm.device)
        return inv

    def expanded_kv_scale(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        return self.kv_head_scale.to(device=device, dtype=torch.float32).repeat_interleave(self.head_dim).to(dtype)

    def expanded_query_scale(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        s = self.kv_head_scale.to(device=device, dtype=torch.float32).repeat_interleave(self.group_size)
        s = s.repeat_interleave(self.head_dim)
        return s.to(dtype)


@dataclass
class LayerGaugeSummary:
    layer_name: str
    kind: str
    size: int
    min_scale: float
    max_scale: float
    median_scale: float


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _row_stat(w: torch.Tensor, *, mode: Literal["mean_abs", "rms", "max_abs"] = "rms") -> torch.Tensor:
    if w.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    w2 = w.to(torch.float32)
    eps = torch.finfo(torch.float32).eps
    if mode == "mean_abs":
        return w2.abs().mean(dim=1) + eps
    if mode == "rms":
        return torch.sqrt(w2.pow(2).mean(dim=1)) + eps
    if mode == "max_abs":
        return w2.abs().amax(dim=1) + eps
    raise ValueError(mode)


def _block_indices(block_perm: torch.Tensor, block_size: int) -> torch.Tensor:
    device = block_perm.device
    base = block_perm.to(torch.int64).unsqueeze(1) * block_size
    offs = torch.arange(block_size, device=device, dtype=torch.int64).unsqueeze(0)
    return (base + offs).reshape(-1)


def _normalize_scales(scales: torch.Tensor, clip: float | None) -> torch.Tensor:
    scales = scales.to(torch.float32)
    gmean = torch.exp(torch.log(scales).mean())
    scales = scales / gmean
    if clip is not None and clip > 1.0:
        scales = torch.clamp(scales, min=1.0 / float(clip), max=float(clip))
    return scales


def _is_local_path(model_id: str) -> bool:
    try:
        return Path(model_id).expanduser().exists()
    except OSError:
        return False


def _normalize_model_id(model_id: str) -> str:
    p = Path(model_id).expanduser()
    if _is_local_path(model_id):
        return str(p.resolve())
    return model_id


# -----------------------------------------------------------------------------
# MLP gauge math
# -----------------------------------------------------------------------------


def _hermite_nodes_weights(order: int, *, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np
    z, w = np.polynomial.hermite.hermgauss(order)
    z = torch.tensor(z, device=device, dtype=dtype)
    w = torch.tensor(w / math.sqrt(math.pi), device=device, dtype=dtype)
    z = z * math.sqrt(2.0)
    return z, w


@torch.no_grad()
def silu_gaussian_moments(sigma: torch.Tensor, *, order: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    device = sigma.device
    dtype = torch.float32
    z, w = _hermite_nodes_weights(order, device=device, dtype=dtype)
    t = sigma.unsqueeze(-1) * z.unsqueeze(0)
    s = F.silu(t)
    m0 = (s.square() * w.unsqueeze(0)).sum(dim=-1)
    m2 = (s.square() * z.square().unsqueeze(0) * w.unsqueeze(0)).sum(dim=-1)
    return m0, m2


@torch.no_grad()
def calibration_free_swiglu_up_gauge_from_weights(
    norm_weight: Optional[torch.Tensor],
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    cfg: GaugeConfig,
) -> SwiGLUUpGauge:
    Din, H = int(w_gate.shape[0]), int(w_gate.shape[1])
    if w_up.shape != (Din, H) or w_down.shape[0] != H:
        raise ValueError("Incompatible SwiGLU shapes")

    device = w_gate.device
    dtype = torch.float32
    eps = torch.finfo(dtype).eps

    if norm_weight is None:
        gamma2 = torch.ones(Din, dtype=dtype, device=device)
    else:
        if norm_weight.numel() != Din:
            raise ValueError(f"norm_weight has {norm_weight.numel()} elements, expected {Din}")
        gamma2 = norm_weight.detach().to(device=device, dtype=dtype).square() + eps

    wg = w_gate.detach().to(device=device, dtype=dtype)
    wu = w_up.detach().to(device=device, dtype=dtype)
    wd = w_down.detach().to(device=device, dtype=dtype)

    sigma_g2 = (wg.square() * gamma2.unsqueeze(1)).sum(dim=0) + eps
    sigma_u2 = (wu.square() * gamma2.unsqueeze(1)).sum(dim=0) + eps
    cov_gu = ((wg * wu) * gamma2.unsqueeze(1)).sum(dim=0)

    sigma_g = sigma_g2.sqrt()
    sigma_u = sigma_u2.sqrt()
    rho = torch.clamp(cov_gu / (sigma_g * sigma_u + eps), min=-0.999, max=0.999)

    m0, m2 = silu_gaussian_moments(sigma_g, order=cfg.hermite_order)
    hidden_rms = sigma_u * torch.sqrt((1.0 - rho.square()) * m0 + rho.square() * m2 + eps)

    a = cfg.lambda_hidden * hidden_rms
    b = cfg.lambda_down * _row_stat(wd, mode=cfg.stat)

    block_size = int(cfg.block_size)
    n_blocks = math.ceil(H / block_size)
    padded = n_blocks * block_size
    if padded != H:
        pad_a = torch.full((padded - H,), float(a.mean()), device=device, dtype=dtype)
        pad_b = torch.full((padded - H,), float(b.mean()), device=device, dtype=dtype)
        a = torch.cat([a, pad_a], dim=0)
        b = torch.cat([b, pad_b], dim=0)

    score = 0.5 * (torch.log(b + eps) - torch.log(a + eps))
    score_blocks = score.view(n_blocks, block_size).mean(dim=1)
    block_perm = torch.argsort(score_blocks)
    perm = _block_indices(block_perm, block_size)[:H]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(H, device=device)

    a_p = a[perm]
    b_p = b[perm]
    a_blk = a_p.view(n_blocks, block_size).mean(dim=1)
    b_blk = b_p.view(n_blocks, block_size).mean(dim=1)
    block_scale = torch.sqrt((b_blk + eps) / (a_blk + eps))
    block_scale = _normalize_scales(block_scale, cfg.scale_clip)

    return SwiGLUUpGauge(
        perm=perm.to(torch.int64),
        inv_perm=inv_perm.to(torch.int64),
        block_scale=block_scale.to(torch.float32),
        block_size=block_size,
    )


@torch.no_grad()
def apply_swiglu_up_gauge(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    gauge: SwiGLUUpGauge,
    *,
    b_gate: Optional[torch.Tensor] = None,
    b_up: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = w_gate.device
    dtype = w_gate.dtype
    H = gauge.perm.numel()
    perm = gauge.perm.to(device=device)
    scale = gauge.expanded_scale(H, device=device, dtype=torch.float32)

    wg_t = w_gate[:, perm]
    wu_t = w_up[:, perm] * scale.unsqueeze(0)
    wd_t = w_down[perm, :] / scale.unsqueeze(1)
    bg_t = None if b_gate is None else b_gate[perm]
    bu_t = None if b_up is None else b_up[perm] * scale

    return wg_t.to(dtype), wu_t.to(dtype), wd_t.to(dtype), (None if bg_t is None else bg_t.to(dtype)), (None if bu_t is None else bu_t.to(dtype))


# -----------------------------------------------------------------------------
# Attention gauge math (Qwen2 text-only)
# -----------------------------------------------------------------------------


@torch.no_grad()
def calibration_free_attention_head_gauge_from_weights(
    norm_weight: Optional[torch.Tensor],
    w_v: torch.Tensor,  # [D_in, H_kv * d_h]
    w_o: torch.Tensor,  # [H_q * d_h, D_out]
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cfg: AttentionGaugeConfig,
) -> AttentionHeadGauge:
    Din = int(w_v.shape[0])
    if w_v.shape[1] != num_kv_heads * head_dim:
        raise ValueError("w_v shape is incompatible with num_kv_heads/head_dim")
    if w_o.shape[0] != num_query_heads * head_dim:
        raise ValueError("w_o shape is incompatible with num_query_heads/head_dim")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("Expected grouped attention with H_q divisible by H_kv")

    g = num_query_heads // num_kv_heads
    device = w_v.device
    dtype = torch.float32
    eps = torch.finfo(dtype).eps

    if norm_weight is None:
        gamma2 = torch.ones(Din, dtype=dtype, device=device)
    else:
        if norm_weight.numel() != Din:
            raise ValueError(f"norm_weight has {norm_weight.numel()} elements, expected {Din}")
        gamma2 = norm_weight.detach().to(device=device, dtype=dtype).square() + eps

    wv = w_v.detach().to(device=device, dtype=dtype)
    wo = w_o.detach().to(device=device, dtype=dtype)

    sigma_v = torch.sqrt((wv.square() * gamma2.unsqueeze(1)).sum(dim=0) + eps)
    sigma_v = sigma_v.view(num_kv_heads, head_dim)
    a_head = cfg.lambda_value * torch.sqrt(sigma_v.square().mean(dim=1) + eps)

    o_row_stat = _row_stat(wo, mode=cfg.stat).view(num_query_heads, head_dim)
    b_head = torch.empty(num_kv_heads, device=device, dtype=dtype)
    for kv in range(num_kv_heads):
        q_lo = kv * g
        q_hi = (kv + 1) * g
        b_head[kv] = o_row_stat[q_lo:q_hi, :].mean() * cfg.lambda_out

    if cfg.permute_heads:
        score = 0.5 * (torch.log(b_head + eps) - torch.log(a_head + eps))
        kv_perm = torch.argsort(score)
    else:
        kv_perm = torch.arange(num_kv_heads, device=device, dtype=torch.int64)
    kv_inv = torch.empty_like(kv_perm)
    kv_inv[kv_perm] = torch.arange(num_kv_heads, device=device)

    if cfg.scale_values:
        a_p = a_head[kv_perm]
        b_p = b_head[kv_perm]
        kv_scale = torch.sqrt((b_p + eps) / (a_p + eps))
        kv_scale = _normalize_scales(kv_scale, cfg.scale_clip)
    else:
        kv_scale = torch.ones(num_kv_heads, device=device, dtype=dtype)

    return AttentionHeadGauge(
        kv_head_perm=kv_perm.to(torch.int64),
        kv_head_inv_perm=kv_inv.to(torch.int64),
        kv_head_scale=kv_scale.to(torch.float32),
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        group_size=g,
    )


@torch.no_grad()
def apply_attention_head_gauge(
    w_q: torch.Tensor,   # [D_in, H_q * d_h]
    w_k: torch.Tensor,   # [D_in, H_kv * d_h]
    w_v: torch.Tensor,   # [D_in, H_kv * d_h]
    w_o: torch.Tensor,   # [H_q * d_h, D_out]
    gauge: AttentionHeadGauge,
    *,
    b_q: Optional[torch.Tensor] = None,
    b_k: Optional[torch.Tensor] = None,
    b_v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = w_q.device
    dtype = w_q.dtype

    q_head_perm = gauge.query_head_perm().to(device=device)
    kv_head_perm = gauge.kv_head_perm.to(device=device)

    idx_q_d = _block_indices(q_head_perm, gauge.head_dim)
    idx_kv_d = _block_indices(kv_head_perm, gauge.head_dim)

    wq_t = w_q[:, idx_q_d]
    bq_t = None if b_q is None else b_q[idx_q_d]

    wk_t = w_k[:, idx_kv_d]
    bk_t = None if b_k is None else b_k[idx_kv_d]

    s_kv = gauge.expanded_kv_scale(device=device, dtype=torch.float32)
    wv_t = w_v[:, idx_kv_d] * s_kv.unsqueeze(0)
    bv_t = None if b_v is None else b_v[idx_kv_d] * s_kv

    s_q = gauge.expanded_query_scale(device=device, dtype=torch.float32)
    wo_t = w_o[idx_q_d, :] / s_q.unsqueeze(1)

    return (
        wq_t.to(dtype),
        wk_t.to(dtype),
        wv_t.to(dtype),
        wo_t.to(dtype),
        (None if bq_t is None else bq_t.to(dtype)),
        (None if bk_t is None else bk_t.to(dtype)),
        (None if bv_t is None else bv_t.to(dtype)),
    )


# -----------------------------------------------------------------------------
# HF / Qwen2 model integration
# -----------------------------------------------------------------------------


def _looks_like_qwen_style_mlp(mlp: nn.Module) -> bool:
    return all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj"))


def _looks_like_qwen2_attention(attn: nn.Module) -> bool:
    return all(hasattr(attn, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))


def _get_int_attr(obj: object, names: tuple[str, ...]) -> int | None:
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return int(val)
    return None


def _resolve_attention_num_q_heads(attn: nn.Module, *, head_dim: int | None, require: bool = True) -> int | None:
    val = _get_int_attr(attn, ("num_heads", "num_attention_heads", "num_q_heads", "num_query_heads", "n_heads"))
    if val is not None:
        return val
    config = getattr(attn, "config", None)
    if config is not None:
        val = _get_int_attr(config, ("num_attention_heads", "num_heads", "num_q_heads", "num_query_heads", "n_heads"))
        if val is not None:
            return val
    if head_dim is not None:
        return int(attn.q_proj.out_features // head_dim)
    if require:
        raise AttributeError("Could not resolve number of query heads for attention module.")
    return None


def _resolve_attention_num_kv_heads(attn: nn.Module, *, head_dim: int | None, require: bool = True) -> int | None:
    val = _get_int_attr(attn, ("num_key_value_heads", "num_kv_heads", "num_kv", "num_key_value", "n_kv_heads"))
    if val is not None:
        return val
    config = getattr(attn, "config", None)
    if config is not None:
        val = _get_int_attr(config, ("num_key_value_heads", "num_kv_heads", "num_kv", "num_key_value", "n_kv_heads"))
        if val is not None:
            return val
    if head_dim is not None:
        return int(attn.v_proj.out_features // head_dim)
    if require:
        raise AttributeError("Could not resolve number of KV heads for attention module.")
    return None


def _resolve_attention_head_dim(attn: nn.Module) -> int:
    val = _get_int_attr(attn, ("head_dim", "head_size", "dim_head"))
    if val is not None:
        return val
    config = getattr(attn, "config", None)
    if config is not None:
        val = _get_int_attr(config, ("head_dim", "head_size", "dim_head"))
        if val is not None:
            return val
    num_q_heads = _resolve_attention_num_q_heads(attn, head_dim=None, require=False)
    if num_q_heads is not None:
        return int(attn.q_proj.out_features // num_q_heads)
    num_kv_heads = _resolve_attention_num_kv_heads(attn, head_dim=None, require=False)
    if num_kv_heads is not None:
        return int(attn.v_proj.out_features // num_kv_heads)
    raise AttributeError("Could not resolve attention head dimension.")


def iter_decoder_layers(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    seen: set[int] = set()
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if id(module) in seen:
            continue
        mlp = getattr(module, "mlp", None)
        attn = getattr(module, "self_attn", None)
        post_ln = getattr(module, "post_attention_layernorm", None)
        input_ln = getattr(module, "input_layernorm", None)
        if mlp is not None and attn is not None and post_ln is not None and input_ln is not None and _looks_like_qwen_style_mlp(mlp) and _looks_like_qwen2_attention(attn):
            seen.add(id(module))
            candidates.append((name, module))

    def sort_key(item: tuple[str, nn.Module]) -> tuple[int, str]:
        name = item[0]
        ints = [int(p) for p in name.split(".") if p.isdigit()]
        return (ints[-1] if ints else 10**9, name)

    for item in sorted(candidates, key=sort_key):
        yield item


@torch.no_grad()
def build_mlp_gauges_calibration_free(model: nn.Module, *, gauge_cfg: GaugeConfig) -> tuple[dict[str, SwiGLUUpGauge], list[LayerGaugeSummary]]:
    gauges: dict[str, SwiGLUUpGauge] = {}
    summaries: list[LayerGaugeSummary] = []
    for layer_name, layer in iter_decoder_layers(model):
        norm_w = getattr(layer.post_attention_layernorm, "weight", None)
        w_gate = layer.mlp.gate_proj.weight.detach().T.contiguous()
        w_up = layer.mlp.up_proj.weight.detach().T.contiguous()
        w_down = layer.mlp.down_proj.weight.detach().T.contiguous()
        gauge = calibration_free_swiglu_up_gauge_from_weights(norm_w, w_gate, w_up, w_down, cfg=gauge_cfg)
        gauges[layer_name] = gauge
        scales = gauge.block_scale.detach().to(torch.float32).cpu()
        summaries.append(LayerGaugeSummary(layer_name=layer_name, kind="mlp", size=int(w_down.shape[0]), min_scale=float(scales.min()), max_scale=float(scales.max()), median_scale=float(scales.median())))
    return gauges, summaries


@torch.no_grad()
def apply_mlp_gauges_in_place(model: nn.Module, gauges: dict[str, SwiGLUUpGauge]) -> None:
    for layer_name, layer in iter_decoder_layers(model):
        gauge = gauges[layer_name]
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj

        w_gate = gate_proj.weight.detach().T.contiguous()
        w_up = up_proj.weight.detach().T.contiguous()
        w_down = down_proj.weight.detach().T.contiguous()
        b_gate = None if gate_proj.bias is None else gate_proj.bias.detach().contiguous()
        b_up = None if up_proj.bias is None else up_proj.bias.detach().contiguous()

        wg_t, wu_t, wd_t, bg_t, bu_t = apply_swiglu_up_gauge(w_gate, w_up, w_down, gauge, b_gate=b_gate, b_up=b_up)

        gate_proj.weight.copy_(wg_t.T.to(gate_proj.weight.dtype))
        up_proj.weight.copy_(wu_t.T.to(up_proj.weight.dtype))
        down_proj.weight.copy_(wd_t.T.to(down_proj.weight.dtype))
        if bg_t is not None and gate_proj.bias is not None:
            gate_proj.bias.copy_(bg_t.to(gate_proj.bias.dtype))
        if bu_t is not None and up_proj.bias is not None:
            up_proj.bias.copy_(bu_t.to(up_proj.bias.dtype))


@torch.no_grad()
def build_attention_gauges_calibration_free(model: nn.Module, *, attn_cfg: AttentionGaugeConfig) -> tuple[dict[str, AttentionHeadGauge], list[LayerGaugeSummary]]:
    gauges: dict[str, AttentionHeadGauge] = {}
    summaries: list[LayerGaugeSummary] = []
    for layer_name, layer in iter_decoder_layers(model):
        attn = layer.self_attn
        norm_w = getattr(layer.input_layernorm, "weight", None)
        head_dim = _resolve_attention_head_dim(attn)
        num_q_heads = _resolve_attention_num_q_heads(attn, head_dim=head_dim)
        num_kv_heads = _resolve_attention_num_kv_heads(attn, head_dim=head_dim)
        w_v = attn.v_proj.weight.detach().T.contiguous()
        w_o = attn.o_proj.weight.detach().T.contiguous()
        gauge = calibration_free_attention_head_gauge_from_weights(
            norm_w,
            w_v,
            w_o,
            num_query_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cfg=attn_cfg,
        )
        gauges[layer_name] = gauge
        scales = gauge.kv_head_scale.detach().to(torch.float32).cpu()
        summaries.append(LayerGaugeSummary(layer_name=layer_name, kind="attn", size=num_kv_heads, min_scale=float(scales.min()), max_scale=float(scales.max()), median_scale=float(scales.median())))
    return gauges, summaries


@torch.no_grad()
def apply_attention_gauges_in_place(model: nn.Module, gauges: dict[str, AttentionHeadGauge]) -> None:
    for layer_name, layer in iter_decoder_layers(model):
        if layer_name not in gauges:
            continue
        attn = layer.self_attn
        gauge = gauges[layer_name]

        q_proj = attn.q_proj
        k_proj = attn.k_proj
        v_proj = attn.v_proj
        o_proj = attn.o_proj

        w_q = q_proj.weight.detach().T.contiguous()
        w_k = k_proj.weight.detach().T.contiguous()
        w_v = v_proj.weight.detach().T.contiguous()
        w_o = o_proj.weight.detach().T.contiguous()
        b_q = None if q_proj.bias is None else q_proj.bias.detach().contiguous()
        b_k = None if k_proj.bias is None else k_proj.bias.detach().contiguous()
        b_v = None if v_proj.bias is None else v_proj.bias.detach().contiguous()

        wq_t, wk_t, wv_t, wo_t, bq_t, bk_t, bv_t = apply_attention_head_gauge(
            w_q, w_k, w_v, w_o, gauge, b_q=b_q, b_k=b_k, b_v=b_v
        )

        q_proj.weight.copy_(wq_t.T.to(q_proj.weight.dtype))
        k_proj.weight.copy_(wk_t.T.to(k_proj.weight.dtype))
        v_proj.weight.copy_(wv_t.T.to(v_proj.weight.dtype))
        o_proj.weight.copy_(wo_t.T.to(o_proj.weight.dtype))
        if bq_t is not None and q_proj.bias is not None:
            q_proj.bias.copy_(bq_t.to(q_proj.bias.dtype))
        if bk_t is not None and k_proj.bias is not None:
            k_proj.bias.copy_(bk_t.to(k_proj.bias.dtype))
        if bv_t is not None and v_proj.bias is not None:
            v_proj.bias.copy_(bv_t.to(v_proj.bias.dtype))


# -----------------------------------------------------------------------------
# Loading / saving
# -----------------------------------------------------------------------------


def load_model_and_tokenizer(model_id: str, *, dtype: torch.dtype = torch.bfloat16, device_map: str | dict | None = None):
    try:
        import transformers
    except Exception as exc:
        raise RuntimeError("This script requires transformers.") from exc

    AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM", None)
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("Could not import AutoTokenizer / AutoModelForCausalLM from transformers.")

    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

    model_id = _normalize_model_id(model_id)
    local_only = _is_local_path(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        device_map=device_map,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def save_outputs(
    model: nn.Module,
    tokenizer,
    save_dir: str | Path,
    *,
    mlp_gauges: dict[str, SwiGLUUpGauge],
    attn_gauges: dict[str, AttentionHeadGauge],
    summaries: list[LayerGaugeSummary],
    model_id: str,
    gauge_cfg: GaugeConfig,
    attn_cfg: AttentionGaugeConfig,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)
    torch.save(mlp_gauges, save_dir / "mlp_gauges_cf_sbgq.pt")
    torch.save(attn_gauges, save_dir / "attn_gauges_cf_sbgq.pt")
    package = {
        "model_id": model_id,
        "mlp_gauge_cfg": asdict(gauge_cfg),
        "attn_gauge_cfg": asdict(attn_cfg),
        "layer_summaries": [asdict(s) for s in summaries],
    }
    (save_dir / "cf_sbgq_qwen25_package.json").write_text(json.dumps(package, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Calibration-free exact SBGQ for Qwen/Qwen2.5 text-only models.")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device-map", default=None)

    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-attn", action="store_true")

    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--stat", choices=["mean_abs", "rms", "max_abs"], default="rms")
    ap.add_argument("--hermite-order", type=int, default=32)
    ap.add_argument("--scale-clip", type=float, default=8.0)

    ap.add_argument("--attn-scale-clip", type=float, default=8.0)
    ap.add_argument("--disable-attn-perm", action="store_true")
    ap.add_argument("--disable-attn-scale", action="store_true")
    return ap


@torch.no_grad()
def main() -> None:
    args = build_arg_parser().parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    mlp_cfg = GaugeConfig(
        block_size=args.block_size,
        lambda_hidden=1.0,
        lambda_down=1.0,
        stat=args.stat,
        hermite_order=args.hermite_order,
        scale_clip=args.scale_clip,
    )
    attn_cfg = AttentionGaugeConfig(
        lambda_value=1.0,
        lambda_out=1.0,
        stat=args.stat,
        scale_clip=args.attn_scale_clip,
        permute_heads=not args.disable_attn_perm,
        scale_values=not args.disable_attn_scale,
    )

    print(f"[QWEN2.5-CF-SBGQ] loading {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, dtype=dtype, device_map=args.device_map)

    summaries: list[LayerGaugeSummary] = []
    mlp_gauges: dict[str, SwiGLUUpGauge] = {}
    attn_gauges: dict[str, AttentionHeadGauge] = {}

    if not args.skip_mlp:
        mlp_gauges, mlp_summaries = build_mlp_gauges_calibration_free(model, gauge_cfg=mlp_cfg)
        apply_mlp_gauges_in_place(model, mlp_gauges)
        summaries.extend(mlp_summaries)
        print(f"[QWEN2.5-CF-SBGQ] applied exact calibration-free MLP gauges to {len(mlp_gauges)} layers")

    if not args.skip_attn:
        attn_gauges, attn_summaries = build_attention_gauges_calibration_free(model, attn_cfg=attn_cfg)
        apply_attention_gauges_in_place(model, attn_gauges)
        summaries.extend(attn_summaries)
        print(f"[QWEN2.5-CF-SBGQ] applied exact calibration-free attention gauges to {len(attn_gauges)} layers")

    for summary in summaries[:8]:
        print(
            f"  - {summary.layer_name} [{summary.kind}] size={summary.size}, "
            f"scale range=[{summary.min_scale:.4f}, {summary.max_scale:.4f}]"
        )
    if len(summaries) > 8:
        print(f"  ... and {len(summaries) - 8} more layer summaries")

    save_outputs(
        model,
        tokenizer,
        args.save_dir,
        mlp_gauges=mlp_gauges,
        attn_gauges=attn_gauges,
        summaries=summaries,
        model_id=args.model_id,
        gauge_cfg=mlp_cfg,
        attn_cfg=attn_cfg,
    )
    print(f"[QWEN2.5-CF-SBGQ] saved outputs to {args.save_dir}")


if __name__ == "__main__":
    main()
