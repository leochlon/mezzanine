from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import random
import math
import time
import os
import inspect


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def call_with_accepted_kwargs(fn, **kwargs):
    try:
        sig = inspect.signature(fn)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**accepted)
    except Exception:
        return fn()


def extract_image(obs: Any, env: Any = None) -> Optional[np.ndarray]:
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] in (3, 4):
        img = obs[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    if isinstance(obs, dict):
        for k in ["image", "rgb", "observation", "obs"]:
            if k in obs:
                return extract_image(obs[k], env=env)
    if env is not None:
        for meth in ["render", "get_image", "get_obs_image", "observe"]:
            if hasattr(env, meth):
                try:
                    return extract_image(getattr(env, meth)(), env=None)
                except Exception:
                    pass
    return None


def default_dt_from_env(env: Any, fallback: float) -> float:
    for attr in ["dt", "delta_t", "interval", "time_step", "step_dt"]:
        if hasattr(env, attr):
            try:
                v = float(getattr(env, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    return fallback


def clickable_indices(positions: Sequence[Any], max_eli: int) -> List[int]:
    no_action_pos = positions[0]
    out = []
    for i in range(1, min(len(positions), max_eli + 1)):
        try:
            if positions[i] != no_action_pos:
                out.append(i)
        except Exception:
            out.append(i)
    return out


@dataclass
class IPhyreCollectConfig:
    games: List[str]
    n_train: int
    n_test: int
    max_steps: int = 260
    delta_seconds: float = 4.0
    sim_dt_fallback: float = 0.1

    # single action per episode
    p_no_action: float = 0.35

    per_episode_samples: int = 4
    max_attempts_factor: int = 12
    seed: int = 0
    log_every_attempts: int = 200
    log_every_seconds: float = 30.0


def collect_iphyre(cfg: IPhyreCollectConfig) -> Dict[str, Any]:
    """Collect (img_t, img_{t+Δ}, action idx+time, action window) from I-PHYRE.

    Uses per-game quotas so multi-game runs are real.
    """
    from iphyre.simulator import IPHYRE
    from iphyre.games import MAX_ELI_OBJ_NUM

    rng = random.Random(cfg.seed)
    max_eli = int(MAX_ELI_OBJ_NUM)

    n_games = max(1, len(cfg.games))
    train_per_game = int(math.ceil(cfg.n_train / n_games))
    test_per_game = int(math.ceil(cfg.n_test / n_games))

    def collect_split(game: str, target: int, split: str) -> List[Dict[str, Any]]:
        env = IPHYRE(game=game)
        positions = env.get_action_space()
        clickables = clickable_indices(positions, max_eli=max_eli)

        dt = default_dt_from_env(env, cfg.sim_dt_fallback)
        delta_steps = max(1, int(round(cfg.delta_seconds / dt)))

        if cfg.max_steps < delta_steps + 2:
            raise ValueError(
                f"max_steps={cfg.max_steps} too small for delta_seconds={cfg.delta_seconds} with dt={dt:.4f} (delta_steps={delta_steps})"
            )

        max_attempts = cfg.max_attempts_factor * target
        got = 0
        attempts = 0
        fail_no_img = 0
        fail_short = 0
        added_total = 0
        attempt_times: List[float] = []
        last_log_t = time.time()

        samples: List[Dict[str, Any]] = []
        print(f"[collect] start game={game} split={split} target={target} dt={dt:.4f}s delta_steps={delta_steps} clickables={len(clickables)} max_attempts={max_attempts}")

        while attempts < max_attempts and got < target:
            attempts += 1
            t_attempt0 = time.time()

            obs0 = call_with_accepted_kwargs(env.reset, use_image=True)
            img0 = extract_image(obs0, env=env)
            if img0 is None:
                fail_no_img += 1
                continue

            action_idxs = [0] * cfg.max_steps
            a_idx = 0
            a_time = 0.0

            do_action = (len(clickables) > 0) and (rng.random() > cfg.p_no_action)
            if do_action:
                a_idx = rng.choice(clickables)
                click_step = rng.randint(0, max(1, min(cfg.max_steps - 1, int(0.6 * delta_steps))))
                action_idxs[click_step] = a_idx
                a_time = float(click_step) / float(max(1, delta_steps - 1))

            frames: List[np.ndarray] = [img0]
            done = False
            for t in range(cfg.max_steps):
                a = action_idxs[t]
                try:
                    obs, r, done = call_with_accepted_kwargs(env.step, positions[a], use_image=True)
                except TypeError:
                    obs, r, done = env.step(positions[a])
                img = extract_image(obs, env=env)
                if img is None:
                    fail_no_img += 1
                    break
                frames.append(img)
                if done:
                    break

            if len(frames) <= delta_steps:
                fail_short += 1
                continue

            possible_t0 = list(range(0, len(frames) - delta_steps))
            rng.shuffle(possible_t0)
            for _ in range(min(cfg.per_episode_samples, len(possible_t0))):
                t0 = possible_t0.pop()
                samples.append({
                    "split": split,
                    "game": game,
                    "dt": float(dt),
                    "delta_steps": int(delta_steps),
                    "img_t": frames[t0],
                    "img_tp": frames[t0 + delta_steps],
                    "action_window": action_idxs[t0: t0 + delta_steps],
                    "action_idx_single": int(a_idx),
                    "action_time_single": float(a_time),
                    "max_eli": int(max_eli),
                })
                got += 1
                added_total += 1
                if got >= target:
                    break

            attempt_times.append(time.time() - t_attempt0)
            if len(attempt_times) > 200:
                attempt_times = attempt_times[-200:]

            now = time.time()
            if (cfg.log_every_attempts > 0 and attempts % cfg.log_every_attempts == 0) or (now - last_log_t) >= cfg.log_every_seconds:
                last_log_t = now
                sample_rate = (got / max(1e-6, sum(attempt_times))) if attempt_times else 0.0
                remaining = max(0, target - got)
                eta_s = (remaining / max(1e-6, sample_rate)) if sample_rate > 0 else float("inf")
                print(f"[collect] progress game={game} split={split} attempts={attempts}/{max_attempts} got={got}/{target} "
                      f"fail_no_img={fail_no_img} fail_short={fail_short} samp/s={sample_rate:.2f} eta={eta_s/60.0:.1f}m")

        if got < target:
            print(f"[collect] warning game={game} split={split}: got {got}/{target} after {attempts} attempts.")
        return samples[:target]

    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for g in cfg.games:
        train.extend(collect_split(g, train_per_game, "train"))
    for g in cfg.games:
        test.extend(collect_split(g, test_per_game, "test"))

    train = train[: cfg.n_train]
    test = test[: cfg.n_test]

    return {
        "train": train,
        "test": test,
        "meta": {
            **cfg.__dict__,
            "train_per_game": train_per_game,
            "test_per_game": test_per_game,
        }
    }


# === Adapter wrapper (for registry + caching) ===

from dataclasses import asdict
from ..core.cache import hash_dict
from .base import WorldAdapter


class IPhyreAdapter(WorldAdapter):
    """Adapter for the I-PHYRE simulator (physics puzzles).

    Produces (img_t, img_{t+Δ}, action features) transitions.
    """

    NAME = "iphyre"
    DESCRIPTION = "I-PHYRE simulator adapter producing image transitions with action metadata."

    def __init__(self, cfg: IPhyreCollectConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def load(self) -> Dict[str, Any]:
        return collect_iphyre(self.cfg)


# Register
from ..registry import ADAPTERS
ADAPTERS.register("iphyre")(IPhyreAdapter)
