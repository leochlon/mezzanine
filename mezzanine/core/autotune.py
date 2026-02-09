from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class TrialResult:
    config: Dict[str, Any]
    metrics: Dict[str, float]


@dataclass
class AutoTuneResult:
    best_config: Optional[Dict[str, Any]]
    trials: List[TrialResult]
    reason: str


class AutoTuner:
    """A tiny 'hard but not dead' pilot-search helper.

    Typical use:
      - generate a small grid of candidate regimes
      - evaluate a cheap baseline metric (difficulty)
      - keep candidates where baseline is within [lo, hi]
      - among those, pick the regime that maximizes an effect size

    This is deliberately generic: recipes can decide what 'baseline' and 'effect' mean.
    """

    def __init__(
        self,
        *,
        baseline_key: str,
        effect_key: str,
        baseline_range: Tuple[float, float],
        maximize_effect: bool = True,
    ):
        self.baseline_key = baseline_key
        self.effect_key = effect_key
        self.baseline_range = baseline_range
        self.maximize_effect = maximize_effect

    def search(
        self,
        grid: Sequence[Dict[str, Any]],
        eval_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> AutoTuneResult:
        trials: List[TrialResult] = []
        lo, hi = self.baseline_range

        best_cfg: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None

        for cfg in grid:
            m = eval_fn(cfg)
            trials.append(TrialResult(config=dict(cfg), metrics=dict(m)))
            base = float(m.get(self.baseline_key, float("nan")))
            eff = float(m.get(self.effect_key, float("nan")))

            ok = (base >= lo) and (base <= hi)
            if not ok:
                continue

            score = eff if self.maximize_effect else -eff
            if best_score is None or score > best_score:
                best_score = score
                best_cfg = dict(cfg)

        if best_cfg is None:
            return AutoTuneResult(best_config=None, trials=trials, reason="No candidates met baseline_range")
        return AutoTuneResult(best_config=best_cfg, trials=trials, reason="Selected by effect among candidates in range")
