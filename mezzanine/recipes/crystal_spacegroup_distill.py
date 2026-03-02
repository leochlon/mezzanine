from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.config import deep_update, load_config
from ..pipelines.regression_distill import (
    MLPRegressorConfig,
    predict,
    train_regressor,
    train_regressor_distill,
    warrant_gap_regression,
)
from ..symmetries.space_group import SpaceGroupConfig, SpaceGroupSymmetry
from ..worlds.matbench_task import MatbenchTaskAdapter, MatbenchTaskAdapterConfig
from .recipe_base import Recipe


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _require_pymatgen():
    try:
        from pymatgen.core import Structure  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "crystal_spacegroup_distill requires optional dependency `pymatgen`."
        ) from e
    return Structure


def _require_matgl_dgl():
    """Import matgl with the DGL backend enabled.

    matgl defaults to the PyG backend; MEGNet/M3GNet/CHGNet live under DGL.
    """

    os.environ.setdefault("MATGL_BACKEND", "DGL")
    try:
        import dgl  # noqa: F401
        import matgl  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Teacher family `matgl_*` requires optional dependencies `matgl` and `dgl`. "
            'Install via: pip install -e ".[materials]" && pip install matgl dgl'
        ) from e


def _matgl_structure_to_graph(structure: Any, *, graph_converter: Any):
    """Convert a pymatgen Structure to a DGLGraph with fields needed for matgl."""
    import torch

    g, lat, state_attr_default = graph_converter.get_graph(structure)
    g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
    g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
    state_attr = torch.tensor(state_attr_default, dtype=torch.float32)
    return g, state_attr


def _matgl_predict_structures(
    model: Any,
    structures: List[Any],
    *,
    graph_converter: Any,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Predict a scalar property for a list of structures with a matgl DGL model."""
    _require_matgl_dgl()
    import dgl
    import torch

    preds: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(structures), int(batch_size)):
            chunk = structures[start : start + int(batch_size)]
            graphs = []
            states = []
            for s in chunk:
                g, st = _matgl_structure_to_graph(s, graph_converter=graph_converter)
                graphs.append(g)
                states.append(st)
            bg = dgl.batch(graphs).to(device)
            st = torch.stack(states, dim=0).to(device)
            y = model(bg, state_attr=st)
            y = y.reshape(-1, 1).detach().cpu().numpy().astype(np.float32)
            preds.append(y)
    return (
        np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float32)
    )


def _train_matgl_regressor(
    model: Any,
    structures_train: List[Any],
    y_train: np.ndarray,
    structures_val: List[Any],
    y_val: np.ndarray,
    *,
    graph_converter: Any,
    steps: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Train a matgl DGL regressor with MSE loss."""
    _require_matgl_dgl()
    import dgl
    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    class _DS(Dataset):
        def __init__(self, structures: List[Any], y: np.ndarray):
            self.structures = structures
            self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            self._cache: List[tuple[Any, Any] | None] = [
                None for _ in range(len(self.structures))
            ]

        def __len__(self) -> int:
            return int(len(self.structures))

        def __getitem__(self, idx: int):
            j = int(idx)
            cached = self._cache[j]
            if cached is None:
                cached = _matgl_structure_to_graph(
                    self.structures[j], graph_converter=graph_converter
                )
                self._cache[j] = cached
            g, st = cached
            return g, st, torch.tensor(self.y[int(idx)], dtype=torch.float32)

    def _collate(batch):
        gs, sts, ys = zip(*batch)
        bg = dgl.batch(list(gs))
        st = torch.stack(list(sts), dim=0)
        yb = torch.stack(list(ys), dim=0)
        return bg, st, yb

    ds = _DS(structures_train, y_train)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        collate_fn=_collate,
    )
    it = iter(dl)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(int(steps)):
        try:
            g, st, yb = next(it)
        except StopIteration:
            it = iter(dl)
            g, st, yb = next(it)
        g = g.to(device)
        st = st.to(device)
        yb = yb.to(device)

        pred = model(g, state_attr=st).reshape(-1, 1)
        loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Val MSE
    yhat_val = _matgl_predict_structures(
        model,
        structures_val,
        graph_converter=graph_converter,
        batch_size=int(batch_size),
        device=device,
    )
    mse_val = float(
        np.mean((yhat_val - np.asarray(y_val, dtype=np.float32).reshape(-1, 1)) ** 2)
    )
    return {"mse_val": mse_val}


def _featurize_structures(
    xs: List[Dict[str, Any]],
    *,
    max_atoms: int,
) -> Tuple[np.ndarray, List[int]]:
    """Featurize pymatgen Structures into a fixed-length vector.

    Returns:
      X: [N_kept, d]
      kept_idx: indices into xs that were kept (n_atoms <= max_atoms)

    Feature layout per structure:
      - lattice parameters (a,b,c, alpha,beta,gamma) => 6
      - per-atom (Z, frac_x, frac_y, frac_z) padded to max_atoms => 4*max_atoms

    IMPORTANT: this featurization is *not* symmetry-invariant by construction;
    that's the point of the experiment.
    """

    _require_pymatgen()

    feats: List[np.ndarray] = []
    kept: List[int] = []
    for i, ex in enumerate(xs):
        s = ex.get("X")
        if s is None:
            continue
        try:
            n = len(s)
        except Exception:
            continue
        if n > int(max_atoms):
            continue

        # Lattice parameters (Angstrom / degrees)
        lat = np.array(list(getattr(s.lattice, "parameters")), dtype=np.float32)

        Z = []
        for site in getattr(s, "sites", []):
            sp = getattr(site, "specie", None) or getattr(site, "species", None)
            # Species can be a Composition; try common fields.
            z = None
            for attr in ("Z", "number"):
                if hasattr(sp, attr):
                    z = getattr(sp, attr)
                    break
            if z is None:
                # Fallback: use string and map is hard; skip.
                z = 0
            Z.append(float(z))
        Z = np.asarray(Z, dtype=np.float32)
        frac = np.asarray(getattr(s, "frac_coords"), dtype=np.float32)

        per_atom = np.concatenate([Z[:, None], frac], axis=1)  # [n,4]
        pad = np.zeros((int(max_atoms), 4), dtype=np.float32)
        pad[:n] = per_atom

        feats.append(np.concatenate([lat.reshape(-1), pad.reshape(-1)], axis=0))
        kept.append(i)

    if not feats:
        return np.zeros((0, 6 + 4 * int(max_atoms)), dtype=np.float32), kept
    return np.stack(feats, axis=0), kept


def _targets(xs: List[Dict[str, Any]], kept_idx: List[int]) -> np.ndarray:
    y = np.array([float(xs[i]["y"]) for i in kept_idx], dtype=np.float32)[:, None]
    return y


def _build_view_augmented_split(
    X_views: np.ndarray,
    y_soft: np.ndarray,
    y_hard: np.ndarray,
    idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten K views into a (K*N_split) dataset with shared labels per structure."""

    if X_views.ndim != 3:
        raise ValueError(f"X_views must be [K,N,d], got {X_views.shape}")
    K, _, d = X_views.shape

    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("idx must be 1D indices")

    if y_soft.ndim == 1:
        y_soft = y_soft[:, None]
    if y_hard.ndim == 1:
        y_hard = y_hard[:, None]
    if y_soft.ndim != 2 or y_hard.ndim != 2:
        raise ValueError("y_soft and y_hard must be [N,D]")
    if y_soft.shape != y_hard.shape:
        raise ValueError("y_soft and y_hard must have the same shape")

    D = int(y_soft.shape[1])
    X_out = X_views[:, idx, :].reshape(-1, int(d))
    y_soft_out = np.repeat(y_soft[idx][None, :, :], int(K), axis=0).reshape(-1, D)
    y_hard_out = np.repeat(y_hard[idx][None, :, :], int(K), axis=0).reshape(-1, D)
    return X_out, y_soft_out, y_hard_out


class CrystalSpaceGroupDistillRecipe(Recipe):
    NAME = "crystal_spacegroup_distill"
    DESCRIPTION = (
        "Crystal property prediction (Matbench): measure space-group warrant gap and distill orbit-averaged "
        "teacher predictions into a student. Requires optional dependencies: matbench + pymatgen."
    )

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        p.add_argument("--dataset_name", type=str, default="matbench_mp_e_form")
        p.add_argument("--fold", type=int, default=0)
        p.add_argument("--n_train", type=int, default=20000)
        p.add_argument("--n_test", type=int, default=5000)

        p.add_argument("--max_atoms", type=int, default=64)
        p.add_argument("--symprec", type=float, default=1e-2)
        p.add_argument("--k_train", type=int, default=4)
        p.add_argument("--k_test", type=int, default=8)

        p.add_argument(
            "--teacher_family",
            type=str,
            default="mlp",
            choices=["mlp", "matgl_megnet", "matgl_m3gnet", "matgl_chgnet"],
            help="Teacher model family. `mlp` uses the naive padded (Z, frac_coords) featurization. "
            "`matgl_*` uses the corresponding DGL model from `matgl` (optional dependency).",
        )
        p.add_argument(
            "--teacher_batch_size",
            type=int,
            default=0,
            help="If >0, overrides batch size used for teacher training/prediction (useful for matgl_*).",
        )

        p.add_argument("--hidden", type=int, default=1024)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=1500)
        p.add_argument("--student_steps", type=int, default=1500)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--wd", type=float, default=1e-4)
        p.add_argument(
            "--hard_label_weight",
            type=float,
            default=0.2,
            help="Student: mix in supervised MSE to true labels (0.0 = pure soft distill; 1.0 = pure supervised).",
        )
        p.add_argument(
            "--student_train_on_views",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="If true, train the student on K symmetry-transformed views per structure, all mapped to the same "
            "orbit-averaged soft label. This is usually required for discrete symmetries like space groups, where the "
            "dataset does not contain random orbit views by default.",
        )

        args = p.parse_args(argv)

        # Apply config file defaults BEFORE seeding/logger/cache creation
        file_cfg = load_config(getattr(args, "config", None))
        merged_cfg = deep_update(file_cfg, self.config)
        self.apply_config_defaults(p, args, merged_cfg)

        ctx = self.build_context(args)
        out_dir = ctx.out_dir
        device = _device()

        adapter_cfg = MatbenchTaskAdapterConfig(
            dataset_name=str(args.dataset_name),
            fold=int(args.fold),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        world = MatbenchTaskAdapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train, kept_tr = _featurize_structures(train, max_atoms=int(args.max_atoms))
        y_train = _targets(train, kept_tr)
        X_test, kept_te = _featurize_structures(test, max_atoms=int(args.max_atoms))
        y_test = _targets(test, kept_te)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            raise ValueError(
                "No usable structures after filtering by max_atoms. "
                "Increase --max_atoms or choose a dataset with smaller cells."
            )

        # Train/val split
        n = int(X_train.shape[0])
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        cfg = MLPRegressorConfig(
            in_dim=int(X_train.shape[1]),
            out_dim=int(y_train.shape[1]),
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )
        teacher_family = str(args.teacher_family)
        teacher_bs = (
            int(args.teacher_batch_size)
            if int(args.teacher_batch_size) > 0
            else int(args.batch_size)
        )

        # Build views by applying random space-group ops to the *kept* structures.
        train_kept = [train[i] for i in kept_tr]
        test_kept = [test[i] for i in kept_te]
        structures_train_kept = [ex["X"] for ex in train_kept]
        structures_test_kept = [ex["X"] for ex in test_kept]

        teacher_device = device
        teacher_graph_converter = None

        if teacher_family == "mlp":
            teacher, teacher_metrics = train_regressor(
                X_train[idx_tr],
                y_train[idx_tr],
                X_train[idx_val],
                y_train[idx_val],
                cfg=cfg,
                steps=int(args.base_steps),
                batch_size=int(teacher_bs),
                lr=float(args.lr),
                wd=float(args.wd),
                device=device,
                seed=int(args.seed),
            )
            yhat_test = predict(teacher, X_test, device=device)
        else:
            _require_matgl_dgl()
            import torch  # type: ignore

            teacher_device = "cuda" if torch.cuda.is_available() else "cpu"

            from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list  # type: ignore
            from matgl.models import CHGNet, MEGNet, M3GNet  # type: ignore

            element_types = get_element_list(structures_train_kept)
            if teacher_family == "matgl_megnet":
                teacher = MEGNet(
                    dropout=float(args.dropout), element_types=element_types
                )
            elif teacher_family == "matgl_m3gnet":
                teacher = M3GNet(
                    dropout=float(args.dropout),
                    element_types=element_types,
                    ntargets=1,
                    task_type="regression",
                )
            elif teacher_family == "matgl_chgnet":
                teacher = CHGNet(
                    element_types=element_types,
                    conv_dropout=float(args.dropout),
                    final_dropout=float(args.dropout),
                    num_targets=1,
                    task_type="regression",
                )
            else:
                raise ValueError(f"Unknown teacher_family: {teacher_family}")

            teacher_graph_converter = Structure2Graph(
                element_types=element_types,
                cutoff=float(getattr(teacher, "cutoff", 5.0)),
            )

            teacher_metrics = _train_matgl_regressor(
                teacher,
                [structures_train_kept[i] for i in idx_tr.tolist()],
                y_train[idx_tr],
                [structures_train_kept[i] for i in idx_val.tolist()],
                y_train[idx_val],
                graph_converter=teacher_graph_converter,
                steps=int(args.base_steps),
                batch_size=int(teacher_bs),
                lr=float(args.lr),
                wd=float(args.wd),
                device=teacher_device,
                seed=int(args.seed),
            )
            yhat_test = _matgl_predict_structures(
                teacher,
                structures_test_kept,
                graph_converter=teacher_graph_converter,
                batch_size=int(teacher_bs),
                device=teacher_device,
            )

        teacher_mse = float(np.mean((yhat_test - y_test) ** 2))

        sym = SpaceGroupSymmetry(
            SpaceGroupConfig(symprec=float(args.symprec), return_operation=False)
        )
        Kt = int(args.k_test)
        preds = []
        for j in range(Kt):
            vj = []
            for i, ex in enumerate(test_kept):
                exj = (
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    if j > 0
                    else ex
                )
                vj.append(exj)
            if teacher_family == "mlp":
                Xj, _ = _featurize_structures(vj, max_atoms=int(args.max_atoms))
                preds.append(predict(teacher, Xj, device=device))
            else:
                if teacher_graph_converter is None:
                    raise RuntimeError(
                        "Internal error: teacher_graph_converter missing for matgl_* teacher."
                    )
                preds.append(
                    _matgl_predict_structures(
                        teacher,
                        [ex2["X"] for ex2 in vj],
                        graph_converter=teacher_graph_converter,
                        batch_size=int(teacher_bs),
                        device=teacher_device,
                    )
                )
        teacher_gap = warrant_gap_regression(np.stack(preds, axis=0))

        # Distill: average teacher predictions over K random space-group ops.
        Kd = int(args.k_train)
        y_soft_views = []
        X_views = [] if bool(args.student_train_on_views) else None
        for j in range(Kd):
            vj = []
            for i, ex in enumerate(train_kept):
                exj = (
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 999, i, j))
                    if j > 0
                    else ex
                )
                vj.append(exj)
            Xj, _ = _featurize_structures(vj, max_atoms=int(args.max_atoms))
            if X_views is not None:
                X_views.append(Xj)
            if teacher_family == "mlp":
                y_soft_views.append(predict(teacher, Xj, device=device))
            else:
                if teacher_graph_converter is None:
                    raise RuntimeError(
                        "Internal error: teacher_graph_converter missing for matgl_* teacher."
                    )
                y_soft_views.append(
                    _matgl_predict_structures(
                        teacher,
                        [ex2["X"] for ex2 in vj],
                        graph_converter=teacher_graph_converter,
                        batch_size=int(teacher_bs),
                        device=teacher_device,
                    )
                )
        y_soft_views_arr = np.stack(y_soft_views, axis=0)  # [K,N,D]
        y_soft = np.mean(y_soft_views_arr, axis=0)  # [N,D]

        if bool(args.student_train_on_views):
            if X_views is None:
                raise RuntimeError(
                    "Internal error: X_views missing despite student_train_on_views=True."
                )
            X_views_arr = np.stack(X_views, axis=0)  # [K,N,d]
            X_tr_s, y_tr_soft, y_tr_hard = _build_view_augmented_split(
                X_views_arr, y_soft, y_train, idx_tr
            )
            X_val_s, y_val_soft, y_val_hard = _build_view_augmented_split(
                X_views_arr, y_soft, y_train, idx_val
            )
        else:
            X_tr_s, X_val_s = X_train[idx_tr], X_train[idx_val]
            y_tr_soft, y_val_soft = y_soft[idx_tr], y_soft[idx_val]
            y_tr_hard, y_val_hard = y_train[idx_tr], y_train[idx_val]

        student, student_metrics = train_regressor_distill(
            X_tr_s,
            y_tr_soft,
            y_tr_hard,
            X_val_s,
            y_val_soft,
            y_val_hard,
            cfg=cfg,
            steps=int(args.student_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            wd=float(args.wd),
            device=device,
            seed=int(args.seed) + 1,
            hard_label_weight=float(args.hard_label_weight),
        )

        yhat_test_s = predict(student, X_test, device=device)
        student_mse = float(np.mean((yhat_test_s - y_test) ** 2))

        preds_s = []
        for j in range(Kt):
            vj = []
            for i, ex in enumerate(test_kept):
                exj = (
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    if j > 0
                    else ex
                )
                vj.append(exj)
            Xj, _ = _featurize_structures(vj, max_atoms=int(args.max_atoms))
            preds_s.append(predict(student, Xj, device=device))
        student_gap = warrant_gap_regression(np.stack(preds_s, axis=0))

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse <= 1.5 * teacher_mse
        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "exp": self.NAME,
            "device": device,
            "world": {
                "adapter": "matbench_task",
                "adapter_config": asdict(adapter_cfg),
                "n_train_raw": len(train),
                "n_test_raw": len(test),
                "n_train_used": int(X_train.shape[0]),
                "n_test_used": int(X_test.shape[0]),
            },
            "symmetry": {"name": "space_group", "symprec": float(args.symprec)},
            "teacher": {
                "family": teacher_family,
                "device": teacher_device,
                "metrics": {**teacher_metrics, "mse_test": teacher_mse, **teacher_gap},
            },
            "student": {
                "family": "mlp",
                "device": device,
                "metrics": {**student_metrics, "mse_test": student_mse, **student_gap},
            },
            "distill": {
                "k_train": Kd,
                "k_test": Kt,
                "max_atoms": int(args.max_atoms),
                "hard_label_weight": float(args.hard_label_weight),
                "student_train_on_views": bool(args.student_train_on_views),
            },
            "make_break": {
                "criterion": {
                    "gap_mse reduces by >=20%": bool(gap_improves),
                    "student mse not worse by >50%": bool(mse_ok),
                },
                "verdict": verdict,
            },
        }

        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        return results
