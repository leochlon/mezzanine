from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class MLPHeadConfig:
    in_dim: int
    num_classes: int
    hidden: int = 512
    depth: int = 1  # 1 = linear, 2 = one hidden layer, ...
    dropout: float = 0.0


class MLPHead(nn.Module):
    def __init__(self, cfg: MLPHeadConfig):
        super().__init__()
        layers = []
        d = cfg.in_dim
        for i in range(max(0, cfg.depth - 1)):
            layers.append(nn.Linear(d, cfg.hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d = cfg.hidden
        layers.append(nn.Linear(d, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_torch(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


def train_hard_label_head(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: MLPHeadConfig,
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[MLPHead, Dict[str, float]]:
    """Train a classifier head on frozen embeddings."""
    torch.manual_seed(seed)
    head = MLPHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    Zt = _to_torch(Z_train, device)
    yt = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
    Zv = _to_torch(Z_val, device)
    yv = torch.from_numpy(y_val).to(device=device, dtype=torch.long)

    n = Zt.shape[0]
    rng = np.random.default_rng(seed)

    head.train()
    pbar = tqdm(range(steps), desc="train(hard)")
    for step in pbar:
        idx = rng.integers(0, n, size=batch_size)
        xb = Zt[idx]
        yb = yt[idx]
        logits = head(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():
                head.eval()
                acc = float((head(Zv).argmax(dim=-1) == yv).float().mean().item())
                head.train()
            pbar.set_postfix(loss=float(loss.item()), val_acc=acc)

    head.eval()
    with torch.no_grad():
        val_logits = head(Zv)
        val_acc = float((val_logits.argmax(dim=-1) == yv).float().mean().item())

    return head, {"val_acc": val_acc}


def train_soft_label_head(
    Z_train: np.ndarray,
    P_teacher: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: MLPHeadConfig,
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
    hard_label_weight: float = 0.0,
    y_train: np.ndarray | None = None,
) -> Tuple[MLPHead, Dict[str, float]]:
    """Train a head to match teacher soft labels on canonical embeddings.

    Loss: KL(teacher || student) implemented as cross-entropy with soft targets.
    Optionally mix in a hard-label CE term (set hard_label_weight > 0).
    """
    torch.manual_seed(seed)
    head = MLPHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    Zt = _to_torch(Z_train, device)
    Pt = _to_torch(P_teacher, device)
    Zv = _to_torch(Z_val, device)
    yv = torch.from_numpy(y_val).to(device=device, dtype=torch.long)

    yt = None
    if hard_label_weight > 0 and y_train is not None:
        yt = torch.from_numpy(y_train).to(device=device, dtype=torch.long)

    n = Zt.shape[0]
    rng = np.random.default_rng(seed)

    head.train()
    pbar = tqdm(range(steps), desc="train(distill)")
    for step in pbar:
        idx = rng.integers(0, n, size=batch_size)
        xb = Zt[idx]
        pb = Pt[idx]
        logits = head(xb)
        logp = F.log_softmax(logits, dim=-1)
        loss = -(pb * logp).sum(dim=-1).mean()

        if hard_label_weight > 0 and yt is not None:
            yb = yt[idx]
            loss_h = F.cross_entropy(logits, yb)
            loss = (1.0 - hard_label_weight) * loss + hard_label_weight * loss_h

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():
                head.eval()
                acc = float((head(Zv).argmax(dim=-1) == yv).float().mean().item())
                head.train()
            pbar.set_postfix(loss=float(loss.item()), val_acc=acc)

    head.eval()
    with torch.no_grad():
        val_logits = head(Zv)
        val_acc = float((val_logits.argmax(dim=-1) == yv).float().mean().item())

    return head, {"val_acc": val_acc}


@torch.no_grad()
def predict_proba(head: MLPHead, Z: np.ndarray, *, device: str = "cuda", batch_size: int = 512) -> np.ndarray:
    head.eval()
    Zt = torch.from_numpy(Z).to(device=device, dtype=torch.float32)
    out = []
    for i in range(0, Zt.shape[0], batch_size):
        logits = head(Zt[i:i+batch_size])
        p = torch.softmax(logits, dim=-1)
        out.append(p.detach().cpu().float().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(axis=1) == y).mean())


def tv_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation distance per example between distributions p and q."""
    return 0.5 * np.abs(p - q).sum(axis=-1)


def warrant_gap_from_views(P_views: np.ndarray) -> Dict[str, float]:
    """Compute warrant-gap style summary stats from symmetry views.

    Args:
        P_views: [N, K, C] probabilities for K symmetry-sampled views.
    Returns:
        dict with:
          - mean_tv_to_mean: E_k TV(p_k, mean_k p_k) averaged over examples
          - mean_pairwise_tv: average over example of TV between two random views
    """
    N, K, C = P_views.shape
    P_bar = P_views.mean(axis=1, keepdims=True)  # [N,1,C]
    tv_to_mean = tv_distance(P_views.reshape(N*K, C), np.repeat(P_bar, K, axis=1).reshape(N*K, C))
    mean_tv_to_mean = float(tv_to_mean.mean())

    # pairwise: pick two views per example deterministically (0,1) if possible else 0,0
    if K >= 2:
        tv_pair = tv_distance(P_views[:, 0, :], P_views[:, 1, :])
        mean_pairwise = float(tv_pair.mean())
    else:
        mean_pairwise = 0.0

    return {"mean_tv_to_mean": mean_tv_to_mean, "mean_pairwise_tv": mean_pairwise}
