from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.cache import LatentCache, LatentCacheConfig, hash_dict
from ..encoders.hf_causal_lm import HFCausalLMEncoder, HFCausalLMEncoderConfig
from ..symmetries.order import OrderSymmetry, OrderSymmetryConfig
from ..pipelines.text_distill import (
    MLPHeadConfig,
    train_soft_label_head,
    predict_proba,
    accuracy,
    warrant_gap_from_views,
)
from ..predictors.hf_causal_lm import HFCausalLMChoicePredictor, HFCausalLMChoicePredictorConfig
from ..viz.text_distill import plot_text_order_distill
from ..worlds.hf_qa import HFQADatasetAdapter, HFQADatasetAdapterConfig
from .recipe_base import Recipe


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    parts = _SENT_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def join_sentences(parts: List[str]) -> str:
    return " ".join([p.strip() for p in parts if p.strip()])


def view_seed(global_seed: int, ex_index: int, view_index: int) -> int:
    # stable 32-bit seed
    x = (global_seed * 1000003 + ex_index * 9176 + view_index * 101) & 0xFFFFFFFF
    return int(x)


def make_boolq_prompts(
    examples: List[Dict[str, Any]],
    *,
    symmetry: OrderSymmetry,
    global_seed: int,
    k: int,
    keep_first_sentences: int = 0,
) -> List[List[str]]:
    """Return K view lists of prompts (each length N)."""
    views: List[List[str]] = [[] for _ in range(k)]
    for i, ex in enumerate(examples):
        passage = str(ex["passage"])
        question = str(ex["question"])
        sents = split_sentences(passage)
        if keep_first_sentences > 0 and len(sents) > keep_first_sentences:
            head = sents[:keep_first_sentences]
            tail = sents[keep_first_sentences:]
        else:
            head, tail = [], sents

        for j in range(k):
            seed = view_seed(global_seed, i, j)
            tail_p = symmetry.sample(tail, seed=seed) if len(tail) > 1 else tail
            passage_j = join_sentences(head + list(tail_p))
            prompt = f"Passage: {passage_j}\nQuestion: {question}\nAnswer:"
            views[j].append(prompt)
    return views


class HFLLMHiddenStateOrderDistill(Recipe):
    """LLM distillation under order symmetry using logits + hidden states.

    Teacher: a frozen causal LM scored on answer choices.
            We define the *warranted* belief as the symmetry-marginalized expectation:
                p*(y|x) = E_{view ~ symmetry}[ p_LM(y | view(x)) ].

    Student: a tiny MLP head on frozen *hidden states* of the same LM.
            Trained on multiple symmetry views with a single target p*(y|x),
            so it becomes invariant to order and approximates p* in one pass.
    """
    NAME = "hf_llm_hiddenstate_order_distill"
    DESCRIPTION = "BoolQ: distill symmetry-marginalized (order-robust) beliefs into a one-pass hidden-state head."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # World
        p.add_argument("--dataset", type=str, default="boolq")
        p.add_argument("--subset", type=str, default=None)
        p.add_argument("--train_split", type=str, default="train")
        p.add_argument("--test_split", type=str, default="validation")
        p.add_argument("--n_train", type=int, default=2048)
        p.add_argument("--n_test", type=int, default=512)

        # Symmetry
        p.add_argument("--k_train", type=int, default=8, help="views per example used to define teacher + train invariance")
        p.add_argument("--k_test", type=int, default=16)
        p.add_argument("--keep_first_sentences", type=int, default=0)

        # Teacher (logits)
        p.add_argument("--lm_name", type=str, default="gpt2")
        p.add_argument("--lm_max_length", type=int, default=512)
        p.add_argument("--lm_batch_size", type=int, default=4)
        p.add_argument("--no_fp16", action="store_true")
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--choices", type=str, default=" yes| no", help="pipe-separated choices; default yes/no with leading space")

        # Hidden-state encoder
        p.add_argument("--embed_layer", type=int, default=-1)
        p.add_argument("--embed_mode", type=str, default="last", choices=["last","mean","mean_std"])
        p.add_argument("--enc_batch_size", type=int, default=8)

        # Head training
        p.add_argument("--train_steps", type=int, default=800)
        p.add_argument("--train_bs", type=int, default=256)
        p.add_argument("--train_lr", type=float, default=1e-3)
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--depth", type=int, default=2)

        # Output
        p.add_argument("--out", type=str, default="results.json")
        p.add_argument("--verbose", action="store_true", help="Print stage/progress logs.")

        args = p.parse_args(argv)
        ctx = self.build_context(args)
        out_dir = ctx.out_dir
        cache = ctx.cache

        def log(msg: str) -> None:
            if args.verbose:
                print(f"[hf_llm_hs] {msg}", flush=True)

        # Load the LM ONCE and share it between teacher + hidden-state encoder to avoid GPU OOM.
        # (Previously this recipe instantiated two separate AutoModelForCausalLM instances.)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = str(args.device)
        use_fp16 = (not bool(args.no_fp16)) and device.startswith("cuda")
        # Prefer bf16 on modern GPUs; fallback to fp16.
        dtype = torch.bfloat16 if use_fp16 else torch.float32
        if use_fp16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            dtype = torch.float16

        log(f"loading tokenizer/lm: name={args.lm_name} device={device}")
        tok = AutoTokenizer.from_pretrained(args.lm_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        log("loading lm weights (this prints 'Loading checkpoint shards'...)")
        model = AutoModelForCausalLM.from_pretrained(
            args.lm_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        log("moving lm to device...")
        model.to(device)
        model.eval()
        log(f"loaded lm={args.lm_name} device={device} dtype={str(dtype).replace('torch.', '')}")

        # Load world
        log(f"loading world dataset={args.dataset} n_train={args.n_train} n_test={args.n_test}")
        t_world = time.time()
        world_cfg = HFQADatasetAdapterConfig(
            dataset=args.dataset,
            subset=args.subset,
            train_split=args.train_split,
            test_split=args.test_split,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
        )
        world = HFQADatasetAdapter(world_cfg)
        data = world.load()
        log(f"world loaded in {time.time() - t_world:.1f}s")
        train_ex = data["train"]
        test_ex = data["test"]
        y_train = np.array([int(ex["label"]) for ex in train_ex], dtype=np.int64)
        y_test = np.array([int(ex["label"]) for ex in test_ex], dtype=np.int64)

        # Symmetry: order (sentence permutation)
        symmetry = OrderSymmetry(OrderSymmetryConfig(keep_first=0))

        # Build views
        k_train = int(args.k_train)
        k_test = int(args.k_test)
        train_views = make_boolq_prompts(train_ex, symmetry=symmetry, global_seed=args.seed, k=k_train, keep_first_sentences=int(args.keep_first_sentences))
        test_views = make_boolq_prompts(test_ex, symmetry=symmetry, global_seed=args.seed + 1, k=k_test, keep_first_sentences=int(args.keep_first_sentences))

        # Teacher predictor (logits/choices)
        choices = [c for c in str(args.choices).split("|") if c != ""]
        if len(choices) < 2:
            raise ValueError("--choices must contain at least 2 options separated by '|'")
        teacher_cfg = HFCausalLMChoicePredictorConfig(
            model_name=args.lm_name,
            batch_size=int(args.lm_batch_size),
            fp16=(not args.no_fp16),
            max_length=int(args.lm_max_length),
            device=args.device,
            progress=bool(args.verbose),
        )
        teacher = HFCausalLMChoicePredictor(teacher_cfg, model=model, tok=tok)
        log(f"teacher ready: choices={choices} bs={teacher_cfg.batch_size} max_len={teacher_cfg.max_length}")

        # Hidden-state encoder
        enc_cfg = HFCausalLMEncoderConfig(
            model_name=args.lm_name,
            batch_size=int(args.enc_batch_size),
            fp16=(not args.no_fp16),
            max_length=int(args.lm_max_length),
            device=args.device,
            layer=int(args.embed_layer),
            embed_mode=str(args.embed_mode),
            progress=bool(args.verbose),
        )
        encoder = HFCausalLMEncoder(enc_cfg, model=model, tok=tok)
        log(f"encoder ready: layer={enc_cfg.layer} mode={enc_cfg.embed_mode} bs={enc_cfg.batch_size} max_len={enc_cfg.max_length}")

        world_fp = world.fingerprint()
        enc_fp = encoder.fingerprint()

        # ---- Teacher: compute p_LM(y | view(x)) for train views ----
        # shape: [N, K, C]
        def _score_views(views: List[List[str]], split: str) -> np.ndarray:
            from tqdm import tqdm
            N = len(views[0])
            K = len(views)
            C = len(choices)
            P = np.zeros((N, K, C), dtype=np.float32)
            for j in tqdm(range(K), desc=f"teacher {split}", disable=not args.verbose):
                t0 = time.time()
                log(f"teacher scoring split={split} view={j+1}/{K} n={N}")
                probs = teacher.predict_proba(views[j], choices=choices)  # [N,C]
                P[:, j, :] = probs
                log(f"teacher done split={split} view={j+1}/{K} in {time.time() - t0:.1f}s")
            return P

        log(f"scoring teacher views: k_train={k_train} k_test={k_test}")
        P_train_views = _score_views(train_views, split="train")
        P_train_teacher = P_train_views.mean(axis=1)  # [N,C]

        # ---- Encode train views into hidden states (with caching per view) ----
        from tqdm import tqdm
        Z_train_views: List[np.ndarray] = []
        for j in tqdm(range(k_train), desc="encode train views", disable=not args.verbose):
            tag = f"llm_hs_train_view{j}"
            if cache is not None:
                key = cache.make_key(world_fingerprint=world_fp, encoder_fingerprint=enc_fp, split="train", tag=tag, extra={"k_train": k_train})
                got = cache.get(key)
                if got is not None:
                    Zj, _meta = got
                    log(f"cache hit: {tag} shape={tuple(Zj.shape)}")
                else:
                    t0 = time.time()
                    log(f"encoding {tag} n={len(train_views[j])}")
                    Zj = encoder.encode(train_views[j])
                    log(f"encoded {tag} in {time.time() - t0:.1f}s shape={tuple(Zj.shape)}")
                    cache.put(key, Zj, meta={"n": len(train_views[j]), "view": j})
            else:
                t0 = time.time()
                log(f"encoding {tag} n={len(train_views[j])}")
                Zj = encoder.encode(train_views[j])
                log(f"encoded {tag} in {time.time() - t0:.1f}s shape={tuple(Zj.shape)}")
            Z_train_views.append(Zj)

        # Flatten train: each view is a training sample
        Z_train = np.concatenate(Z_train_views, axis=0)  # [N*K, D]
        P_teacher_flat = np.repeat(P_train_teacher, repeats=k_train, axis=0)  # [N*K, C]

        # Validation embeddings: use test view0
        if cache is not None:
            key = cache.make_key(world_fingerprint=world_fp, encoder_fingerprint=enc_fp, split="test", tag="llm_hs_test_view0", extra={"k_test": k_test})
            got = cache.get(key)
            if got is not None:
                Z_val, _meta = got
                log(f"cache hit: llm_hs_test_view0 shape={tuple(Z_val.shape)}")
            else:
                t0 = time.time()
                log(f"encoding llm_hs_test_view0 n={len(test_views[0])}")
                Z_val = encoder.encode(test_views[0])
                log(f"encoded llm_hs_test_view0 in {time.time() - t0:.1f}s shape={tuple(Z_val.shape)}")
                cache.put(key, Z_val, meta={"n": len(test_views[0]), "view": 0})
        else:
            t0 = time.time()
            log(f"encoding llm_hs_test_view0 n={len(test_views[0])}")
            Z_val = encoder.encode(test_views[0])
            log(f"encoded llm_hs_test_view0 in {time.time() - t0:.1f}s shape={tuple(Z_val.shape)}")

        # Train head
        log(f"training head: steps={args.train_steps} bs={args.train_bs} lr={args.train_lr}")
        head_cfg = MLPHeadConfig(in_dim=int(Z_train.shape[1]), num_classes=len(choices), hidden=int(args.hidden), depth=int(args.depth))
        head, head_metrics = train_soft_label_head(
            Z_train=Z_train,
            P_teacher=P_teacher_flat,
            Z_val=Z_val,
            y_val=y_test,
            cfg=head_cfg,
            steps=int(args.train_steps),
            batch_size=int(args.train_bs),
            lr=float(args.train_lr),
            device=args.device,
            seed=int(args.seed),
        )
        log("head training done")

        # ---- Evaluate on test views ----
        log("scoring teacher test views...")
        P_test_views = _score_views(test_views, split="test")
        gap_base = warrant_gap_from_views(P_test_views)
        acc_base_v0 = accuracy(P_test_views[:, 0, :], y_test)
        acc_base_mean = accuracy(P_test_views.mean(axis=1), y_test)

        # Student: apply head to each view's hidden states
        P_stud_views_list: List[np.ndarray] = []
        for j in tqdm(range(k_test), desc="encode test views", disable=not args.verbose):
            # cache per view
            tag = f"llm_hs_test_view{j}"
            if cache is not None:
                key = cache.make_key(world_fingerprint=world_fp, encoder_fingerprint=enc_fp, split="test", tag=tag, extra={"k_test": k_test})
                got = cache.get(key)
                if got is not None:
                    Zj, _meta = got
                    log(f"cache hit: {tag} shape={tuple(Zj.shape)}")
                else:
                    t0 = time.time()
                    log(f"encoding {tag} n={len(test_views[j])}")
                    Zj = encoder.encode(test_views[j])
                    log(f"encoded {tag} in {time.time() - t0:.1f}s shape={tuple(Zj.shape)}")
                    cache.put(key, Zj, meta={"n": len(test_views[j]), "view": j})
            else:
                t0 = time.time()
                log(f"encoding {tag} n={len(test_views[j])}")
                Zj = encoder.encode(test_views[j])
                log(f"encoded {tag} in {time.time() - t0:.1f}s shape={tuple(Zj.shape)}")

            Pj = predict_proba(head, Zj, device=args.device)
            P_stud_views_list.append(Pj)

        P_stud_views = np.stack(P_stud_views_list, axis=1)  # [N,K,C]
        gap_stud = warrant_gap_from_views(P_stud_views)
        acc_stud_v0 = accuracy(P_stud_views[:, 0, :], y_test)
        acc_stud_mean = accuracy(P_stud_views.mean(axis=1), y_test)

        # Teacher mixture (expensive, K forward passes)
        P_teacher_test = P_test_views.mean(axis=1)  # [N,C]
        acc_teacher = accuracy(P_teacher_test, y_test)

        # Student-to-teacher agreement
        tv_student_to_teacher = float(
            (0.5 * np.abs(P_stud_views - P_teacher_test[:, None, :]).sum(axis=2)).mean()
        )

        # Make/break criterion: reduce instability without hurting acc too much
        rel_gap = gap_stud["mean_tv_to_mean"] / max(1e-9, gap_base["mean_tv_to_mean"])
        make = (rel_gap <= 0.70) and (acc_stud_v0 >= acc_base_v0 - 0.05)

        summary: Dict[str, Any] = {
            "exp": self.NAME,
            "world": asdict(world_cfg),
            "meta": data.get("meta", {}),
            "choices": choices,
            "k_train": k_train,
            "k_test": k_test,
            "encoder": {"name": "hf_causal_lm", "cfg": asdict(enc_cfg)},
            "teacher": {"name": "hf_causal_lm_choice", "cfg": asdict(teacher_cfg)},
            "head": {"cfg": asdict(head_cfg), "metrics": head_metrics},
            "baseline": {
                "acc": float(acc_base_v0),
                "acc_mean_views": float(acc_base_mean),
                "gap_mean_tv_to_mean": float(gap_base["mean_tv_to_mean"]),
                "gap_pairwise_tv": float(gap_base["mean_pairwise_tv"]),
            },
            "teacher_mixture": {
                "acc": float(acc_teacher),
                "note": "This is the symmetry-marginalized expectation E_view[p_LM(y|view(x))]. Requires K forward passes at inference.",
            },
            "student": {
                "acc": float(acc_stud_v0),
                "acc_mean_views": float(acc_stud_mean),
                "gap_mean_tv_to_mean": float(gap_stud["mean_tv_to_mean"]),
                "gap_pairwise_tv": float(gap_stud["mean_pairwise_tv"]),
                "tv_to_teacher_mean": float(tv_student_to_teacher),
            },
            "make_break": {
                "criterion": {
                    "gap_reduction": "student_gap <= 0.70 * baseline_gap",
                    "acc_drop": "student_acc >= baseline_acc - 0.05",
                },
                "rel_gap": float(rel_gap),
                "verdict": "MAKE ✅" if make else "BREAK / INCONCLUSIVE ❌",
            },
        }

        # write outputs
        out_path = out_dir / str(args.out)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        # simple diagnostic plot (reuses text_distill plot expectations)
        try:
            fig_path = out_dir / "diagnostics.png"
            plot_text_order_distill({"baseline": {"acc": summary["baseline"]["acc"], "gap_mean_tv_to_mean": summary["baseline"]["gap_mean_tv_to_mean"]},
                                     "student": {"acc": summary["student"]["acc"], "gap_mean_tv_to_mean": summary["student"]["gap_mean_tv_to_mean"]}}, fig_path)
        except Exception:
            pass

        return summary


__all__ = ["HFLLMHiddenStateOrderDistill"]
