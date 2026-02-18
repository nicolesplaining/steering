"""
Gisting: Compressing behavior descriptions into learned gist tokens.

Based on "Learning to Compress Prompts with Gist Tokens" (Mu et al., 2023).

Compresses long metacognitive behavior descriptions (~200-400 tokens) into k
learned gist tokens via a custom attention mask that creates an information
bottleneck.

Training sequence:
  [system+behavior_text | GIST_0..k | user_problem | assistant_solution]
  - Gist tokens attend to behavior text (standard causal)
  - Problem/solution tokens CANNOT attend to behavior text (masked)
  - Loss only on solution tokens
  - Gist tokens learn to compress behavior information

Inference:
  1. Forward pass: behavior_text + gist_tokens -> extract gist KV cache
  2. Custom generation: gist_cache + problem -> solution
  RoPE-correct: problem tokens start at position n_hint + k

Usage:
  python gisting.py --phase data              # Build dataset
  python gisting.py --phase train --k 4       # Train gist tokens + LoRA
  python gisting.py --phase eval              # Evaluate gist condition
  python gisting.py --phase probe             # Quick feasibility (data+train+eval)
  python gisting.py --phase verify            # Run verification checks
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.cache_utils import DynamicCache
from peft import LoraConfig, TaskType, get_peft_model

from baseline_eval import extract_boxed_answer
from hint_and_rerun import (
    _format_behaviors_as_lines,  # COUPLING: if this changes format, retrain
    build_hinted_prompt,
    is_correct,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SYSTEM_MESSAGE = "You are a helpful math assistant."

DEFAULT_INPUTS = [
    "results/run_20260203_052646/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json",
    "results/run_20260203_064154/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING — behavior-format only
# ═══════════════════════════════════════════════════════════════════════════════


def load_behavior_data(
    paths: list[str],
    *,
    require_correct: bool = True,
) -> list[dict]:
    """
    Load behavior-format _with_hints.json files and extract training triples.

    Only loads files with config.strong_behaviors=True.
    Deduplicates by problem text across files.

    Returns list of dicts with keys:
      hint_text, hint_raw, problem, teacher_solution, ground_truth, source_file
    """
    dataset: list[dict] = []
    seen_problems: set[str] = set()

    for path in paths:
        if not os.path.exists(path):
            print(f"  Skipping missing: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)

        config = blob.get("config", {})
        if not config.get("strong_behaviors"):
            print(f"  Skipping {path} (not strong_behaviors)")
            continue
        if config.get("mock_hints") or config.get("use_solution_hint"):
            print(f"  Skipping {path} (mock/solution hints)")
            continue

        items = blob.get("results", [])
        loaded = 0

        for item in items:
            problem = (item.get("problem") or "").strip()
            if not problem:
                continue

            rerun = item.get("rerun") or {}
            hint_raw = item.get("hint") or ""
            teacher_response = rerun.get("response") or ""

            if not hint_raw or not teacher_response:
                continue

            if require_correct and not rerun.get("correct"):
                continue

            # Dedup AFTER filtering so correct items from later files aren't blocked
            if problem in seen_problems:
                continue
            seen_problems.add(problem)

            hint_formatted = _format_behaviors_as_lines(hint_raw)

            dataset.append({
                "hint_text": hint_formatted,
                "hint_raw": hint_raw,
                "problem": problem,
                "teacher_solution": teacher_response,
                "ground_truth": (
                    rerun.get("ground_truth")
                    or item.get("original", {}).get("ground_truth", "")
                ),
                "source_file": os.path.abspath(path),
            })
            loaded += 1

        print(f"  {path}: {loaded} examples ({len(items)} total in file)")

    print(f"  Total: {len(dataset)} examples")
    return dataset


def split_dataset(
    data: list[dict],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split into train/eval. Returns (train, eval)."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_train = int(len(data) * train_frac)
    train = [data[i] for i in indices[:n_train]]
    eval_ = [data[i] for i in indices[n_train:]]
    print(f"  Split: {len(train)} train, {len(eval_)} eval")
    return train, eval_


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def build_hint_prefix_ids(tokenizer, behavior_text: str) -> list[int]:
    """
    Build token IDs for the hint region (everything before gist tokens).

    Layout: <|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n{behavior_text}

    Must be reproduced EXACTLY at inference for gist cache to be valid.
    """
    text = (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"{behavior_text}"
    )
    return tokenizer.encode(text, add_special_tokens=False)


def build_problem_suffix_ids(tokenizer, problem: str) -> list[int]:
    """
    Build token IDs for the problem region (after gist tokens).

    Matches build_hinted_prompt() strong_behaviors format, but WITHOUT the
    actual behavior lines (those are in the hint prefix, compressed by gist).
    """
    user_content = (
        f"Problem: {problem}\n\n"
        "A behavior is a note or skill to keep in mind while solving math "
        "problems. It can be a strategy, a trick, or a technique. It can also "
        "be a general rule or a common sense principle. The behavior is not a "
        "solution to the problem, but it can be used to solve the problem. "
        "You must apply the behaviors above. In your reasoning, explicitly "
        "reference the behavior names when you use them (e.g., behavior_x). "
        "Avoid vague explanations; show the key intermediate step each "
        "behavior enables. Please reason step by step and put the final "
        "answer in \\boxed{}."
    )
    text = (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return tokenizer.encode(text, add_special_tokens=False)


def build_solution_ids(tokenizer, solution_text: str) -> list[int]:
    """Build token IDs for the solution region."""
    # Strip trailing <|im_end|> if present (we add it ourselves)
    clean = solution_text.rstrip()
    if clean.endswith("<|im_end|>"):
        clean = clean[: -len("<|im_end|>")].rstrip()
    text = f"{clean}<|im_end|>"
    return tokenizer.encode(text, add_special_tokens=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GIST ATTENTION MASK
# ═══════════════════════════════════════════════════════════════════════════════


def build_gist_mask(
    n_hint: int,
    k_gist: int,
    n_problem: int,
    n_solution: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build 4D attention mask for gist training.

    Sequence layout: [hint | gist | problem | solution]

    Rules:
      hint tokens:     standard causal within themselves
      gist tokens:     attend to ALL hint + previous gist (causal)
      problem tokens:  attend to gist + previous problem (BLOCKED from hint)
      solution tokens: attend to gist + all problem + previous solution (BLOCKED from hint)

    Returns (1, 1, total, total) float mask. 0.0 = attend, -inf = blocked.
    """

    total = n_hint + k_gist + n_problem + n_solution
    # Standard causal mask (True = can attend)
    mask = torch.tril(torch.ones(total, total, dtype=torch.bool, device=device))
    # Block post-gist tokens from attending to hint tokens
    post_gist_start = n_hint + k_gist
    mask[post_gist_start:, :n_hint] = False
    # Convert: 0.0 = attend, -inf = blocked (HuggingFace additive convention)
    float_mask = torch.where(mask, 0.0, torch.finfo(dtype).min)
    return float_mask.unsqueeze(0).unsqueeze(0)


def verify_gist_mask():
    """Print a small mask for visual verification."""
    n_h, k, n_p, n_s = 3, 2, 3, 3
    mask = build_gist_mask(n_h, k, n_p, n_s, torch.float32, torch.device("cpu"))
    mask_2d = mask.squeeze()

    labels = (
        [f"h{i}" for i in range(n_h)]
        + [f"g{i}" for i in range(k)]
        + [f"p{i}" for i in range(n_p)]
        + [f"s{i}" for i in range(n_s)]
    )

    print("\nGist Mask (1=attend, .=blocked):")
    header = "      " + " ".join(f"{l:>3}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = mask_2d[i]
        cells = ["  1" if row[j] == 0.0 else "  ." for j in range(len(row))]
        print(f" {row_label:>4} " + " ".join(cells))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL SETUP
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GistConfig:
    model_name: str = DEFAULT_MODEL
    k: int = 4
    lr: float = 2e-4
    epochs: int = 5
    grad_accum_steps: int = 8
    max_seq_len: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_dropout: float = 0.05
    seed: int = 42
    dataset_path: str = "gist_dataset.json"
    adapter_path: str = "gist_adapter"


def setup_gist_model(cfg: GistConfig):
    """
    Load base model with eager attention, add gist tokens, attach LoRA.

    Eager attention is required because Flash Attention 2 does NOT support
    arbitrary 4D masks, and SDPA may silently dispatch to Flash.

    Returns (model, tokenizer, gist_token_ids).
    """

    print(f"  Loading {cfg.model_name} (attn_implementation='eager')...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Verify no sliding window on any layers
    sw = getattr(model.config, "use_sliding_window", None)
    mwl = getattr(model.config, "max_window_layers", None)
    nhl = getattr(model.config, "num_hidden_layers", None)
    if sw and mwl is not None and nhl is not None and mwl < nhl:
        print(f"  WARNING: sliding window active (max_window_layers={mwl} < "
              f"num_hidden_layers={nhl}). Gist mask may not work on all layers.")

    # Add gist tokens
    # Design choice: k distinct gist tokens vs. official repo's single <GIST> repeated.
    # Distinct tokens: each position gets its own embedding (more expressive).
    # Single token: relies on RoPE to differentiate positions (fewer params).
    # For small k=4, either works. We use distinct tokens.
    gist_tokens = [f"<GIST_{i}>" for i in range(cfg.k)]
    tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})
    model.resize_token_embeddings(len(tokenizer))
    gist_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in gist_tokens]
    print(f"  Added {cfg.k} gist tokens: ids {gist_token_ids}")

    # Initialize gist embeddings as mean of existing embeddings (Mu et al. 2023)
    # https://github.com/jayelm/gisting/blob/main/src/train.py#L186-L204
    with torch.no_grad():
        embed = model.model.embed_tokens
        mean_embed = embed.weight[:-cfg.k].mean(dim=0)
        for gid in gist_token_ids:
            embed.weight[gid] = mean_embed
        if not getattr(model.config, "tie_word_embeddings", True):
            mean_lm = model.lm_head.weight[:-cfg.k].mean(dim=0)
            for gid in gist_token_ids:
                model.lm_head.weight[gid] = mean_lm
    print(f"  Initialized gist embeddings as mean of existing embeddings")

    # Attach LoRA — try Option C (trainable_token_indices), fall back to A
    model, method = _attach_lora(model, cfg, gist_token_ids, len(tokenizer))
    print(f"  LoRA attached ({method})")
    model.print_trainable_parameters()
    return model, tokenizer, gist_token_ids


def _attach_lora(model, cfg, gist_token_ids, vocab_size):
    """Try Option C (trainable_token_indices), fall back to Option A (hook)."""
    # Option C: PEFT trainable_token_indices
    try:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            trainable_token_indices=gist_token_ids,
        )
        model = get_peft_model(model, lora_cfg)
        # Verify embedding requires grad
        embed = model.base_model.model.model.embed_tokens
        if not embed.weight.requires_grad:
            raise RuntimeError("Embedding not trainable")
        return model, "Option C: trainable_token_indices"
    except (TypeError, RuntimeError) as e:
        print(f"  Option C failed ({e}), trying Option A...")

    # Option A: gradient hook to zero non-gist rows
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    embed = model.base_model.model.model.embed_tokens
    embed.weight.requires_grad_(True)

    gist_mask = torch.zeros(vocab_size, dtype=torch.bool, device=embed.weight.device)
    for gid in gist_token_ids:
        gist_mask[gid] = True

    def _zero_nongist_grad(grad):
        grad[~gist_mask] = 0.0
        return grad

    embed.weight.register_hook(_zero_nongist_grad)
    return model, "Option A: gradient hook"


def load_gist_model(cfg: GistConfig):
    """
    Load a trained gist model from checkpoint for inference/eval.

    Uses setup_gist_model to create the same LoRA structure as training
    (same vocab size, same gist tokens, same LoRA config), then loads the
    trained weights into it via load_adapter + set_adapter. This avoids
    the double-wrap bug of unwrapping and re-wrapping with PeftModel.

    Returns (model, tokenizer, gist_token_ids).
    """
    model, tokenizer, gist_token_ids = setup_gist_model(cfg)

    # Load trained LoRA weights into the existing LoRA structure
    adapter_config_path = os.path.join(cfg.adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        model.load_adapter(cfg.adapter_path, adapter_name="trained")
        model.set_adapter("trained")
        print(f"  Loaded LoRA weights from {cfg.adapter_path}")
    else:
        print(f"  WARNING: No adapter found at {cfg.adapter_path}, using fresh weights")

    # Load trained gist token embeddings
    embed_path = os.path.join(cfg.adapter_path, "gist_embeddings.pt")
    if os.path.exists(embed_path):
        ckpt = torch.load(embed_path, map_location="cpu", weights_only=True)
        embed = model.base_model.model.model.embed_tokens
        saved_ids = ckpt["gist_token_ids"]
        saved_embeds = ckpt["gist_embeddings"].to(embed.weight.device)
        embed.weight.data[saved_ids] = saved_embeds
        print(f"  Loaded gist embeddings from {embed_path}")
        if "lm_head_embeddings" in ckpt:
            lm_head = model.base_model.model.lm_head
            lm_head.weight.data[saved_ids] = ckpt["lm_head_embeddings"].to(
                lm_head.weight.device
            )
            print(f"  Loaded lm_head gist embeddings from {embed_path}")

    return model, tokenizer, gist_token_ids


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATASET
# ═══════════════════════════════════════════════════════════════════════════════


class GistDataset(Dataset):
    """
    Dataset for gist training. batch_size=1 (no collation complexity).

    Each example:
      input_ids:      [hint | gist | problem | solution]
      attention_mask:  (1, 1, L, L) 4D gist mask
      labels:          -100 except on solution tokens
      n_hint:          hint region length (for RoPE at inference)
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        gist_token_ids: list[int],
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.gist_token_ids = gist_token_ids
        self.k = len(gist_token_ids)

        self.examples: list[dict] = []
        skipped = 0
        for item in data:
            hint_ids = build_hint_prefix_ids(tokenizer, item["hint_text"])
            problem_ids = build_problem_suffix_ids(tokenizer, item["problem"])
            solution_ids = build_solution_ids(tokenizer, item["teacher_solution"])

            total_len = len(hint_ids) + self.k + len(problem_ids) + len(solution_ids)
            if total_len > max_seq_len:
                max_sol = max_seq_len - len(hint_ids) - self.k - len(problem_ids)
                if max_sol < 32:
                    skipped += 1
                    continue
                solution_ids = solution_ids[:max_sol]

            self.examples.append({
                "hint_ids": hint_ids,
                "problem_ids": problem_ids,
                "solution_ids": solution_ids,
                "n_hint": len(hint_ids),
                "n_problem": len(problem_ids),
                "n_solution": len(solution_ids),
            })

        if skipped:
            print(f"  GistDataset: skipped {skipped}/{len(data)} (too long)")
        print(f"  GistDataset: {len(self.examples)} examples ready")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        input_ids = (
            ex["hint_ids"]
            + self.gist_token_ids
            + ex["problem_ids"]
            + ex["solution_ids"]
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Labels: -100 everywhere except solution tokens
        labels = torch.full_like(input_ids, -100)
        sol_start = ex["n_hint"] + self.k + ex["n_problem"]
        labels[sol_start:] = input_ids[sol_start:]

        # 4D gist mask (built on CPU, moved to device in training loop)
        mask = build_gist_mask(
            n_hint=ex["n_hint"],
            k_gist=self.k,
            n_problem=ex["n_problem"],
            n_solution=ex["n_solution"],
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": mask,
            "n_hint": ex["n_hint"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_gist(
    model,
    tokenizer,
    gist_token_ids: list[int],
    train_data: list[dict],
    cfg: GistConfig,
):
    """Train gist tokens + LoRA. batch_size=1 to avoid 4D mask collation."""

    ds = GistDataset(train_data, tokenizer, gist_token_ids, cfg.max_seq_len)
    if len(ds) == 0:
        print("  ERROR: No training examples. Check data filters.")
        return model
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    total_steps = max(1, (len(loader) // cfg.grad_accum_steps) * cfg.epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, max(1, total_steps // 10)),
        num_training_steps=total_steps,
    )

    print(f"  Training: {len(ds)} examples, {cfg.epochs} epochs, "
          f"accum={cfg.grad_accum_steps}, total_steps={total_steps}")

    model.train()
    global_step = 0

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            mask = batch["attention_mask"].to(model.device, dtype=torch.bfloat16)

            out = model(
                input_ids=input_ids,
                attention_mask=mask,
                labels=labels,
            )

            loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            epoch_loss += out.loss.item()
            n_batches += 1

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % 20 == 0:
                avg = epoch_loss / n_batches
                print(f"    Epoch {epoch+1} | step {step+1}/{len(loader)} "
                      f"| loss {avg:.4f}")

        # Flush remaining gradients (optimizer only, no scheduler step to
        # avoid advancing LR beyond total_steps)
        if len(loader) % cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1} done — avg loss {avg_loss:.4f}")

    # Save adapter + gist embeddings
    os.makedirs(cfg.adapter_path, exist_ok=True)
    model.save_pretrained(cfg.adapter_path)

    embed = model.base_model.model.model.embed_tokens
    gist_embeds = embed.weight.data[gist_token_ids].cpu()
    save_dict = {
        "gist_token_ids": gist_token_ids,
        "gist_embeddings": gist_embeds,
        "k": len(gist_token_ids),
        "gist_tokens": [f"<GIST_{i}>" for i in range(len(gist_token_ids))],
    }
    # Save lm_head rows for non-tied models
    if not getattr(model.base_model.model.config, "tie_word_embeddings", True):
        lm_head = model.base_model.model.lm_head
        save_dict["lm_head_embeddings"] = lm_head.weight.data[gist_token_ids].cpu()
    torch.save(save_dict, os.path.join(cfg.adapter_path, "gist_embeddings.pt"))
    print(f"  Saved adapter + gist embeddings -> {cfg.adapter_path}/")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE — RoPE-Correct Gist Cache
# ═══════════════════════════════════════════════════════════════════════════════


def _sample_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Sample a single token with temperature and nucleus (top-p) filtering."""
    logits = logits[:, -1, :]  # (batch=1, vocab)
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Zero out tokens beyond the top-p nucleus
    mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[mask] = float("-inf")
    # Scatter back to original order and sample
    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def compute_gist_cache(
    model,
    tokenizer,
    behavior_text: str,
    gist_token_ids: list[int],
) -> tuple[DynamicCache, int]:
    """
    Forward hint_prefix + gist_tokens, extract gist-only KV cache.

    Returns (gist_cache, n_hint). n_hint is needed for RoPE-correct position
    computation: problem tokens must start at position n_hint + k.

    This uses the default causal mask (no custom gist mask), matching the
    official gisting implementation (compress.py passes all-ones attention_mask_gist).
    The gist mask is only needed during training to force the information bottleneck.
    At inference, all tokens attend normally; we just extract and keep the gist KV entries.
    """
    hint_ids = build_hint_prefix_ids(tokenizer, behavior_text)
    prefix_ids = hint_ids + list(gist_token_ids)
    input_ids = torch.tensor([prefix_ids], device=model.device)

    n_hint = len(hint_ids)
    k = len(gist_token_ids)

    out = model(input_ids=input_ids, use_cache=True)

    # Extract ONLY gist positions from the cache
    full_cache = out.past_key_values
    gist_cache = DynamicCache()

    for layer_idx in range(len(full_cache.key_cache)):
        key_states = full_cache.key_cache[layer_idx][:, :, -k:, :].clone()
        value_states = full_cache.value_cache[layer_idx][:, :, -k:, :].clone()
        gist_cache.update(key_states, value_states, layer_idx)

    return gist_cache, n_hint


@torch.no_grad()
def generate_with_gist(
    model,
    tokenizer,
    problem: str,
    gist_cache: DynamicCache,
    n_hint: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> str:
    """
    Generate solution using gist KV cache with correct RoPE positions.

    Cannot use model.generate() because it auto-computes position_ids from
    past_key_values.get_seq_length(), returning k + len(problem) instead of
    n_hint + k + len(problem). Every token after the first would have RoPE
    off by n_hint positions, producing gibberish.

    Sampling matches the original metacog pipeline (temperature=0.6, top_p=0.95).
    """
    k = gist_cache.key_cache[0].shape[2]
    problem_ids = build_problem_suffix_ids(tokenizer, problem)
    input_ids = torch.tensor([problem_ids], device=model.device)

    # Problem tokens start at correct RoPE position
    start_pos = n_hint + k
    seq_len = len(problem_ids)
    position_ids = torch.arange(
        start_pos, start_pos + seq_len, device=model.device
    ).unsqueeze(0)
    cache_position = torch.arange(
        start_pos, start_pos + seq_len, device=model.device
    )

    # Deep copy gist cache (generation mutates it in-place)
    working_cache = DynamicCache()
    for layer_idx in range(len(gist_cache.key_cache)):
        working_cache.update(
            gist_cache.key_cache[layer_idx].clone(),
            gist_cache.value_cache[layer_idx].clone(),
            layer_idx,
        )

    # First forward: all problem tokens
    out = model(
        input_ids=input_ids,
        past_key_values=working_cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
    )

    # Autoregressive decoding with temperature sampling
    next_pos = start_pos + seq_len
    next_token = _sample_token(out.logits[:, -1:], temperature, top_p)
    generated = [next_token.item()]
    cache = out.past_key_values

    eos_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {eos_id, im_end_id} - {None}

    for _ in range(max_new_tokens - 1):
        if generated[-1] in stop_ids:
            break

        out = model(
            input_ids=next_token,
            past_key_values=cache,
            position_ids=torch.tensor([[next_pos]], device=model.device),
            cache_position=torch.tensor([next_pos], device=model.device),
            use_cache=True,
        )
        next_token = _sample_token(out.logits[:, -1:], temperature, top_p)
        generated.append(next_token.item())
        cache = out.past_key_values
        next_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_gist(
    model,
    tokenizer,
    gist_token_ids: list[int],
    eval_data: list[dict],
    max_new_tokens: int = 2048,
) -> dict:
    """
    Evaluate gist compression on held-out examples.

    For context: eval examples come from problems where the model originally
    got the answer wrong (baseline = 0%) but got it right with full behavior
    text in context (full-hint ~ 100% by construction). So gist accuracy > 0%
    means the information bottleneck is working.
    """
    model.eval()
    results = []
    correct_count = 0

    for i, item in enumerate(eval_data):
        gist_cache, n_hint = compute_gist_cache(
            model, tokenizer, item["hint_text"], gist_token_ids
        )

        response = generate_with_gist(
            model, tokenizer, item["problem"],
            gist_cache, n_hint, max_new_tokens,
        )

        predicted = extract_boxed_answer(response)
        gt = item.get("ground_truth", "")
        is_correct_flag, debug = is_correct(predicted, gt)
        match_type = debug.get("match_type", "unknown")

        if is_correct_flag:
            correct_count += 1

        results.append({
            "problem": item["problem"][:100],
            "predicted": predicted,
            "ground_truth": gt,
            "correct": is_correct_flag,
            "match_type": match_type,
            "response_len": len(response),
        })

        status = "OK" if is_correct_flag else "X"
        if (i + 1) % 5 == 0 or i == len(eval_data) - 1:
            acc = correct_count / (i + 1)
            print(f"    [{i+1}/{len(eval_data)}] [{status}] "
                  f"acc={acc:.1%} pred={predicted[:30]}")

    total = len(eval_data)
    accuracy = correct_count / total if total else 0.0
    print(f"\n  Gist eval: {correct_count}/{total} correct")
    if total < 20:
        print(f"  WARNING: only {total} eval examples — report raw counts, "
              f"not percentages. 1/{total} vs 0/{total} is noise.")
    print(f"  accuracy={accuracy:.1%} (baseline=0%, full-hint~100% by construction)")
    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VERIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def verify_embedding_gradients(model, gist_token_ids: list[int], vocab_size: int):
    """After one backward pass, check that only gist rows have gradients."""
    embed = model.base_model.model.model.embed_tokens

    if embed.weight.grad is None:
        print("  WARNING: No gradients on embedding (run backward first)")
        return False

    grad = embed.weight.grad
    gist_grad_norm = grad[gist_token_ids].norm().item()

    non_gist = [i for i in range(min(100, vocab_size)) if i not in gist_token_ids]
    non_gist_grad_norm = grad[non_gist].norm().item() if non_gist else 0.0

    print(f"  Gist row grad norm: {gist_grad_norm:.6f}")
    print(f"  Non-gist sample grad norm: {non_gist_grad_norm:.6f}")

    ok = gist_grad_norm > 0 and non_gist_grad_norm < 1e-10
    print(f"  Embedding gradient check: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_chat_template(tokenizer, gist_token_ids: list[int]):
    """Print tokenized training sequence for visual inspection."""
    behavior_text = "behavior_test: This is a test behavior."
    problem = "What is 2+2?"
    solution = "The answer is \\boxed{4}."

    hint_ids = build_hint_prefix_ids(tokenizer, behavior_text)
    problem_ids = build_problem_suffix_ids(tokenizer, problem)
    solution_ids = build_solution_ids(tokenizer, solution)

    print("\n=== Training Sequence Layout ===")
    print(f"Hint region ({len(hint_ids)} tokens):")
    print(f"  '{tokenizer.decode(hint_ids)}'")
    print(f"Gist tokens ({len(gist_token_ids)} tokens):")
    print(f"  '{tokenizer.decode(gist_token_ids)}'")
    print(f"Problem region ({len(problem_ids)} tokens):")
    print(f"  '{tokenizer.decode(problem_ids)}'")
    print(f"Solution region ({len(solution_ids)} tokens):")
    print(f"  '{tokenizer.decode(solution_ids)}'")
    total = len(hint_ids) + len(gist_token_ids) + len(problem_ids) + len(solution_ids)
    print(f"\nTotal: {total} tokens")
    print("=" * 40)


def run_verification(cfg: GistConfig):
    """Run all verification checks."""
    model, tokenizer, gist_token_ids = setup_gist_model(cfg)

    print("\n1. Mask verification:")
    verify_gist_mask()

    print("2. Chat template verification:")
    verify_chat_template(tokenizer, gist_token_ids)

    print("\n3. Forward pass test with custom 4D mask:")
    hint_ids = build_hint_prefix_ids(tokenizer, "behavior_test: Test.")
    problem_ids = build_problem_suffix_ids(tokenizer, "What is 2+2?")
    solution_ids = build_solution_ids(tokenizer, "The answer is \\boxed{4}.")

    full_ids = hint_ids + list(gist_token_ids) + problem_ids + solution_ids
    input_ids = torch.tensor([full_ids], device=model.device)
    labels = torch.full_like(input_ids, -100)
    sol_start = len(hint_ids) + len(gist_token_ids) + len(problem_ids)
    labels[0, sol_start:] = input_ids[0, sol_start:]

    mask = build_gist_mask(
        len(hint_ids), len(gist_token_ids), len(problem_ids), len(solution_ids),
        torch.bfloat16, model.device,
    )

    out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
    print(f"  Forward pass OK. Loss: {out.loss.item():.4f}, "
          f"logits shape: {out.logits.shape}")

    print("\n4. Backward pass + embedding gradient check:")
    out.loss.backward()
    verify_embedding_gradients(model, gist_token_ids, len(tokenizer))

    print("\n5. Gist cache extraction test:")
    model.eval()
    gist_cache, n_hint = compute_gist_cache(
        model, tokenizer, "behavior_test: Test.", gist_token_ids
    )
    k = len(gist_token_ids)
    n_layers = len(gist_cache.key_cache)
    kv_shape = gist_cache.key_cache[0].shape
    print(f"  Gist cache: {n_layers} layers, shape per layer: {kv_shape}")
    print(f"  n_hint={n_hint}, k={k}")

    print("\n6. RoPE-correct generation test:")
    response = generate_with_gist(
        model, tokenizer, "What is 2+2?",
        gist_cache, n_hint, max_new_tokens=64,
    )
    print(f"  Generated ({len(response)} chars): {response[:200]}")
    coherent = len(response) > 5 and not response.startswith("<")
    print(f"  Coherence check: {'PASS (non-empty)' if coherent else 'WARN (short/garbled)'}")

    print("\nAll verification checks complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gisting: compress behaviors into learned gist tokens"
    )
    parser.add_argument(
        "--phase",
        choices=["data", "train", "eval", "probe", "verify", "all"],
        default="probe",
    )
    parser.add_argument(
        "--input", action="append", default=None,
        help="Input _with_hints.json file(s). Can pass multiple.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of gist tokens.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", default="gist_dataset.json")
    parser.add_argument("--adapter-path", default="gist_adapter")
    parser.add_argument(
        "--require-correct", action="store_true", default=True,
        help="Only use examples where rerun was correct (default: True).",
    )
    parser.add_argument(
        "--no-require-correct", action="store_false", dest="require_correct",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = GistConfig(
        model_name=args.model,
        k=args.k,
        lr=args.lr,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        dataset_path=args.dataset_path,
        adapter_path=args.adapter_path,
    )

    input_paths = args.input or DEFAULT_INPUTS

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── verify ──
    if args.phase == "verify":
        run_verification(cfg)
        return

    # ── data ──
    train_data = eval_data = None
    if args.phase in ("data", "probe", "all"):
        print("\n[Phase: data] Loading behavior-format data...")
        data = load_behavior_data(
            input_paths, require_correct=args.require_correct
        )
        if not data:
            print("  No data loaded! Check input paths and filters.")
            return
        train_data, eval_data = split_dataset(data, train_frac=0.8, seed=cfg.seed)
        with open(cfg.dataset_path, "w") as f:
            json.dump({"train": train_data, "eval": eval_data}, f, indent=2)
        print(f"  Saved dataset -> {cfg.dataset_path}")
        if args.phase == "data":
            return

    # ── train ──
    model = tokenizer = gist_token_ids = None
    if args.phase in ("train", "probe", "all"):
        # Load dataset if needed
        if train_data is None:
            with open(cfg.dataset_path) as f:
                ds = json.load(f)
            train_data = ds["train"]
            eval_data = ds["eval"]

        print("\n[Phase: train] Setting up gist model...")
        model, tokenizer, gist_token_ids = setup_gist_model(cfg)

        print("\n[Phase: train] Training...")
        model = train_gist(model, tokenizer, gist_token_ids, train_data, cfg)

        if args.phase == "train":
            return

    # ── eval ──
    if args.phase in ("eval", "probe", "all"):
        # Load model/data if needed
        if eval_data is None:
            with open(cfg.dataset_path) as f:
                ds = json.load(f)
            eval_data = ds["eval"]

        if model is None:
            print("\n[Phase: eval] Loading trained gist model...")
            model, tokenizer, gist_token_ids = load_gist_model(cfg)

        print(f"\n[Phase: eval] Evaluating gist (k={cfg.k})...")
        gist_results = evaluate_gist(
            model, tokenizer, gist_token_ids, eval_data
        )

        # Save results
        results_path = os.path.join(
            os.path.dirname(cfg.adapter_path) or ".",
            "gist_eval_results.json",
        )
        with open(results_path, "w") as f:
            json.dump({
                "config": {
                    "k": cfg.k,
                    "model": cfg.model_name,
                    "adapter_path": cfg.adapter_path,
                    "n_train": len(train_data) if train_data else "unknown",
                    "n_eval": len(eval_data),
                },
                "gist": gist_results,
            }, f, indent=2)
        print(f"  Results -> {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
