from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _l2norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


@dataclass
class LatentDynamicsTrainConfig:
    steps: int = 800
    batch: int = 256
    lr: float = 1e-3
    wd: float = 1e-4
    hidden: int = 1024
    depth: int = 2
    seed: int = 0


def _mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Module:
    layers = []
    cur = in_dim
    for _ in range(depth):
        layers += [nn.Linear(cur, hidden), nn.GELU()]
        cur = hidden
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


def train_latent_dynamics(
    z_t: np.ndarray,
    z_tp: np.ndarray,
    a_feat: np.ndarray,
    cfg: LatentDynamicsTrainConfig,
    device: str,
) -> Dict[str, Any]:
    """Train two predictors:
        - action-conditioned: (z_t, a_feat) -> z_{t+Δ}
        - no-action: z_t -> z_{t+Δ}

    Loss: cosine distance in embedding space.
    """
    torch.manual_seed(cfg.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)

    zt = torch.tensor(z_t, dtype=torch.float32)
    ztp = torch.tensor(z_tp, dtype=torch.float32)
    a = torch.tensor(a_feat, dtype=torch.float32)

    ds = TensorDataset(zt, ztp, a)
    dl = DataLoader(ds, batch_size=cfg.batch, shuffle=True, drop_last=True)

    d = z_t.shape[1]
    ad = a_feat.shape[1]

    m_a = _mlp(d + ad, d, cfg.hidden, cfg.depth).to(device)
    m_n = _mlp(d, d, cfg.hidden, cfg.depth).to(device)

    opt_a = torch.optim.AdamW(m_a.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    opt_n = torch.optim.AdamW(m_n.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    def cos_loss(pred, tgt):
        pred = torch.nn.functional.normalize(pred, dim=-1)
        tgt = torch.nn.functional.normalize(tgt, dim=-1)
        return 1.0 - (pred * tgt).sum(dim=-1).mean()

    it = iter(dl)
    for s in range(cfg.steps):
        try:
            bt, btp, ba = next(it)
        except StopIteration:
            it = iter(dl)
            bt, btp, ba = next(it)

        bt = bt.to(device); btp = btp.to(device); ba = ba.to(device)

        pa = m_a(torch.cat([bt, ba], dim=-1))
        la = cos_loss(pa, btp)
        opt_a.zero_grad(set_to_none=True)
        la.backward()
        opt_a.step()

        pn = m_n(bt)
        ln = cos_loss(pn, btp)
        opt_n.zero_grad(set_to_none=True)
        ln.backward()
        opt_n.step()

    return {"model_action": m_a, "model_noact": m_n}


def eval_latent_dynamics(
    models: Dict[str, Any],
    z_t: np.ndarray,
    z_tp: np.ndarray,
    a_feat: np.ndarray,
    device: str,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate with:
      - cosine similarity to true future latent
      - retrieval ranks (mean rank, median rank, R@1, R@10)
      - action shuffle counterfactual
    """
    rng = np.random.default_rng(seed)

    z_true = _l2norm(z_tp.astype(np.float32))
    z_persist = _l2norm(z_t.astype(np.float32))

    bt = torch.tensor(z_t, dtype=torch.float32, device=device)
    at = torch.tensor(a_feat, dtype=torch.float32, device=device)

    with torch.no_grad():
        zn = torch.nn.functional.normalize(models["model_noact"](bt), dim=-1).detach().cpu().float().numpy()
        za = torch.nn.functional.normalize(models["model_action"](torch.cat([bt, at], dim=-1)), dim=-1).detach().cpu().float().numpy()
        perm = rng.permutation(len(a_feat))
        zas = torch.nn.functional.normalize(models["model_action"](torch.cat([bt, at[perm]], dim=-1)), dim=-1).detach().cpu().float().numpy()

    def mean_cos(pred):
        return float(np.mean(np.sum(pred * z_true, axis=1)))

    pool = z_true

    def ranks(pred):
        sims = pred @ pool.T
        true = np.diag(sims)
        r = (sims > true[:, None]).sum(axis=1) + 1
        return r

    r_p = ranks(z_persist)
    r_n = ranks(zn)
    r_a = ranks(za)
    r_s = ranks(zas)

    def r_at_k(r, k): return float(np.mean(r <= k))

    metrics = {
        "cos": {
            "persist": mean_cos(z_persist),
            "no_action": mean_cos(zn),
            "action": mean_cos(za),
            "action_shuf": mean_cos(zas),
        },
        "rank": {
            "mean": {
                "persist": float(np.mean(r_p)),
                "no_action": float(np.mean(r_n)),
                "action": float(np.mean(r_a)),
                "action_shuf": float(np.mean(r_s)),
            },
            "median": {
                "persist": float(np.median(r_p)),
                "no_action": float(np.median(r_n)),
                "action": float(np.median(r_a)),
                "action_shuf": float(np.median(r_s)),
            },
            "r@1": {
                "persist": r_at_k(r_p, 1),
                "no_action": r_at_k(r_n, 1),
                "action": r_at_k(r_a, 1),
                "action_shuf": r_at_k(r_s, 1),
            },
            "r@10": {
                "persist": r_at_k(r_p, 10),
                "no_action": r_at_k(r_n, 10),
                "action": r_at_k(r_a, 10),
                "action_shuf": r_at_k(r_s, 10),
            },
        },
    }

    deltas = {
        "action_minus_no_action": {
            "cos": metrics["cos"]["action"] - metrics["cos"]["no_action"],
            "mean_rank_gain": metrics["rank"]["mean"]["no_action"] - metrics["rank"]["mean"]["action"],
            "r10_gain": metrics["rank"]["r@10"]["action"] - metrics["rank"]["r@10"]["no_action"],
        },
        "action_minus_action_shuf": {
            "cos": metrics["cos"]["action"] - metrics["cos"]["action_shuf"],
            "mean_rank_gain": metrics["rank"]["mean"]["action_shuf"] - metrics["rank"]["mean"]["action"],
            "r10_gain": metrics["rank"]["r@10"]["action"] - metrics["rank"]["r@10"]["action_shuf"],
        },
        "action_minus_persist": {
            "cos": metrics["cos"]["action"] - metrics["cos"]["persist"],
            "mean_rank_gain": metrics["rank"]["mean"]["persist"] - metrics["rank"]["mean"]["action"],
            "r10_gain": metrics["rank"]["r@10"]["action"] - metrics["rank"]["r@10"]["persist"],
        },
    }

    # Make/break criteria (rank-based)
    action_helps = (deltas["action_minus_no_action"]["mean_rank_gain"] >= 5.0) or (deltas["action_minus_no_action"]["r10_gain"] >= 0.05)
    shuffle_hurts = (deltas["action_minus_action_shuf"]["mean_rank_gain"] >= 5.0) or (deltas["action_minus_action_shuf"]["r10_gain"] >= 0.05)
    verdict = "MAKE ✅" if (action_helps and shuffle_hurts) else "BREAK / INCONCLUSIVE ❌"

    return {
        "metrics": metrics,
        "deltas": deltas,
        "make_break": {
            "criterion": {
                "action helps vs no-action (mean-rank gain ≥5 OR R@10 gain ≥0.05)": action_helps,
                "action-shuffle hurts (mean-rank gain ≥5 OR R@10 gain ≥0.05)": shuffle_hurts,
            },
            "verdict": verdict,
        }
    }
