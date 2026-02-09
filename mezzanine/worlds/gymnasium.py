from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.cache import hash_dict
from ..core.deterministic import seed_everything
from .base import WorldAdapter


@dataclass
class GymnasiumAdapterConfig:
    env_id: str = "CartPole-v1"
    n_train_episodes: int = 50
    n_test_episodes: int = 10
    max_steps: int = 200
    seed: int = 0
    render_rgb: bool = True


class GymnasiumAdapter(WorldAdapter):
    """Gymnasium adapter (simulators as worlds).

    Provides trajectories of observations/actions/rewards. If render_rgb=True,
    attempts to store rgb frames (env.render()).

    Note: requires gymnasium installed and the env to support rgb_array rendering.
    """
    NAME = "gymnasium"
    DESCRIPTION = "Gymnasium environment adapter (trajectories as worlds)."

    def __init__(self, cfg: GymnasiumAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def _rollout(self, env: Any, max_steps: int, seed: int) -> Dict[str, Any]:
        obs, _ = env.reset(seed=seed)
        frames: List[np.ndarray] = []
        obs_list: List[Any] = []
        act_list: List[Any] = []
        rew_list: List[float] = []
        done = False

        for t in range(max_steps):
            if self.cfg.render_rgb:
                try:
                    fr = env.render()
                    if isinstance(fr, np.ndarray):
                        frames.append(fr)
                except Exception:
                    pass
            obs_list.append(obs)
            a = env.action_space.sample()
            act_list.append(a)
            obs, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            rew_list.append(float(r))
            if done:
                break
        return {"obs": obs_list, "actions": act_list, "rewards": rew_list, "frames": frames}

    def load(self) -> Dict[str, Any]:
        try:
            import gymnasium as gym  # type: ignore
        except Exception as e:
            raise RuntimeError("Gymnasium adapter requires gymnasium. Install: pip install mezzanine[gym]") from e

        seed_everything(self.cfg.seed)
        render_mode = "rgb_array" if self.cfg.render_rgb else None
        env = gym.make(self.cfg.env_id, render_mode=render_mode)

        train = [self._rollout(env, self.cfg.max_steps, self.cfg.seed + i) for i in range(self.cfg.n_train_episodes)]
        test = [self._rollout(env, self.cfg.max_steps, self.cfg.seed + 1000 + i) for i in range(self.cfg.n_test_episodes)]
        meta = {"env_id": self.cfg.env_id, "seed": self.cfg.seed, "max_steps": self.cfg.max_steps}
        try:
            env.close()
        except Exception:
            pass
        return {"train": train, "test": test, "meta": meta}


# Register
from ..registry import ADAPTERS
ADAPTERS.register("gymnasium")(GymnasiumAdapter)
