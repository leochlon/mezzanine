# LLM-oriented examples

Mezzanine was designed to be **cross-domain**: language, vision, robotics, physics, agent trajectories.

The key abstraction is the same everywhere:

1) Choose a **world** (data / environment).
2) Choose a **symmetry family** (order / view / factorization / action-shuffle).
3) **Measure** instability of beliefs or representations (the *warrant gap*).
4) **Distill** the symmetry-marginalized expectation into a single forward pass.

Below are a few *language-first* examples.

## Example A: text order symmetry distillation (encoder + head)

This is the simplest "LLM-ish" demo that runs fast and is very reproducible:

- World: HF dataset (e.g., `ag_news`, `imdb`)
- Symmetry: **sentence order** (shuffle sentences)
- Backbone: frozen HF model used as an **encoder** (BERT/DistilBERT/*or a decoder-only LLM*)
- Distillation: student head matches the **average** prediction over sentence permutations

CLI:
```bash
mezzanine run hf_text_order_distill --out out_text       --dataset ag_news --n_train 5000 --n_test 2000       --model_name distilbert-base-uncased       --k_train 8 --k_test 16
```

Swap in a decoder-only "LLM encoder" (works best with a padding token set):
```bash
mezzanine run hf_text_order_distill --out out_text_llm       --dataset ag_news --n_train 2000 --n_test 1000       --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0       --pool mean --max_length 256
```

## Example B: prompt-level order sensitivity (causal LM scoring)

For generative models, one robust way to define a "belief distribution" is:

- fix a prompt template
- define a small set of discrete answers (choices)
- score each choice by log-probability under the LM

Warrant gap then becomes: how much do those answer probabilities change under
symmetry transformations of the *evidence section* of the prompt (sentence order, paraphrase, etc.).

See: `examples/llm_boolq_order_gap.py` for a minimal reference implementation.

## Example C: distill an LLM's symmetry-marginalized belief into a tiny student

Once you can compute the symmetry-marginalized teacher distribution:
\[
    p_T(y \mid x) = \mathbb{E}_{\eta \sim \mathcal{N}}\big[p_{\text{LM}}(y \mid \eta(x))\big],
\]
you can train a student that uses a single canonical view:
\[
    \min_\theta\; \mathrm{KL}\big(p_T(\cdot \mid x)\;\Vert\;p_\theta(\cdot \mid x)\big).
\]

The package recipe `hf_text_order_distill` is exactly this pattern (with an encoder+head),
and the same objective works when the teacher is a causal LM (prompt scoring).


## Example D: *strong* LLM distillation (logits → hidden-state head)

This is the recommended "LLM distill" demo if you want something that is:

- genuinely **LLM-native** (teacher comes from **logits**),
- genuinely **representation-based** (student uses **hidden states**, not text-only probing),
- still lightweight (frozen backbone + tiny head).

World: BoolQ (passage + question → yes/no)\
Symmetry: sentence order in the passage\
Teacher: frozen causal LM scoring `{yes,no}` via log-prob\
Student: MLP head on frozen **hidden states** trained to match the teacher's **symmetry-marginalized expectation**

```bash
mezzanine run hf_llm_hiddenstate_order_distill --out out_llm_hs \
  --dataset boolq --n_train 2048 --n_test 512 \
  --lm_name gpt2 --embed_layer -1 --embed_mode last \
  --k_train 8 --k_test 16
```

What to look for:

- **Baseline** (single view) shows a sizable warrant gap under sentence permutations.
- **Teacher mixture** (expensive, K views) defines the symmetry-marginalized belief.
- **Student** reduces gap substantially and stays close to the teacher mixture, but runs in **one pass**.
