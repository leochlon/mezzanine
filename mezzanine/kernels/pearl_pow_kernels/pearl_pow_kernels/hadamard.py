from __future__ import annotations

import math
from typing import Tuple

import torch


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def fwht(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Fast Walsh–Hadamard Transform (FWHT) along `dim`.

    This implements multiplication by the **unnormalized** Hadamard matrix H_n
    where H_n^2 = n I (for n a power of two).

    Notes
    -----
    - Works with float and complex dtypes.
    - Autograd-friendly (uses basic tensor ops; no in-place ops).
    - For randomized Hadamard rotations, the unnormalized H is convenient: H^{-1} = H / n.

    Parameters
    ----------
    x:
        Input tensor.
    dim:
        Dimension to transform.

    Returns
    -------
    torch.Tensor
        Transformed tensor with same shape as `x`.
    """
    n = x.shape[dim]
    if not is_power_of_two(int(n)):
        raise ValueError(f"FWHT length must be power-of-two, got n={n}")

    if n == 1:
        return x

    # Move dim to last for easier reshaping.
    y = x.movedim(dim, -1).contiguous()
    orig_shape = y.shape
    # Flatten all leading dims into one batch dim.
    y = y.reshape(-1, n)

    h = 1
    while h < n:
        # Shape: [batch, n/(2h), 2, h]
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        y = y.reshape(-1, n)
        h *= 2

    y = y.reshape(orig_shape)
    y = y.movedim(-1, dim)
    return y


def fwht_rows(x: torch.Tensor) -> torch.Tensor:
    """FWHT over rows (dim=1) for a 2D matrix."""
    if x.ndim != 2:
        raise ValueError("fwht_rows expects a 2D tensor")
    return fwht(x, dim=1)


def fwht_cols(x: torch.Tensor) -> torch.Tensor:
    """FWHT over columns (dim=0) for a 2D matrix."""
    if x.ndim != 2:
        raise ValueError("fwht_cols expects a 2D tensor")
    return fwht(x, dim=0)
