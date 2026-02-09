"""Mezzanine: measure the warrant gap, distill symmetry-marginalized invariants.

Core claim (operationalized):
  *Finite reasoners can be Bayes-optimal in expectation yet brittle under changes in
  view/order/factorization.* Mezzanine makes that brittleness measurable (the *warrant gap*)
  and provides a reusable distillation pattern that turns many-view expectations into a
  single forward-pass invariant state.

Key components:
  - registries : adapters / symmetries / encoders / recipes discoverable via CLI
  - caching    : disk cache for latents keyed by (world fingerprint, encoder fingerprint)
  - autotune   : pilot-search for "hard but not dead" regimes
  - configs    : YAML/JSON configs + deterministic subsampling + global seeding
  - logging    : optional W&B / TensorBoard hooks (never required)
"""

__all__ = ["__version__", "plugins", "registry", "core", "api", "measure"]
__version__ = "1.2.0"

from .api import measure
