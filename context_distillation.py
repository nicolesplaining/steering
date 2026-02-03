"""
Context Distillation Pipeline
Based on "Learning by Distilling Context" (Snell et al., 2022)

Core loop (Section 2.2):
    1.  Sample x ~ D                          (raw task input)
    2.  Teacher generates y ~ P(· | T_teacher(x))   (scratchpad + answer)
    3.  Extract final answer:  f(y)
    4.  Fine-tune student to predict f(y) | T_student(x)

Usage
-----
# 1) Generate the distillation dataset only (cheap, no GPU needed):
    python context_distillation.py --phase dataset --n 500

# 2) Full pipeline — dataset + LoRA fine-tune:
    python context_distillation.py --phase all --n 2000 --student-model meta-llama/Llama-3.2-3B-Instruct

# 3) Import and define your own task:
    from context_distillation import TaskDef, run_pipeline
    my_task = TaskDef(input_distribution=..., teacher_template=..., ...)
    run_pipeline(task=my_task)

# 4) (Project-specific) Build a dataset directly from existing hint_and_rerun
#    outputs (no online teacher calls):
    from context_distillation import (
        build_dataset_from_hint_outputs,
        FinetuneConfig,
        finetune_student,
    )
    ds = build_dataset_from_hint_outputs([
        "results/run_hints5_20260128_090143/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json",
    ])
    model, tok = finetune_student(ds, FinetuneConfig(model_name="Qwen/Qwen2.5-Math-1.5B-Instruct"))
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ─── optional LoRA dep; hard-fail early if missing ───────────────────────────
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    raise ImportError(
        "peft is required for LoRA fine-tuning.  pip install peft"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TASK DEFINITION  –  the four components from Section 2.1
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TaskDef:
    """
    Captures the four design-time choices the paper requires:

        D               – samples a raw task input string
        T_teacher       – wraps that input with rich context (instructions, examples, CoT prompt)
        T_student       – wraps with minimal context (what the student sees after distillation)
        f               – extracts the final answer from the teacher's full output
    """

    input_distribution: Callable[[], str]
    teacher_template: Callable[[str], str]
    student_template: Callable[[str], str]
    answer_extractor: Callable[[str], str]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DEMO TASKS
# ═══════════════════════════════════════════════════════════════════════════════


# ── 2a.  Addition with scratchpad (Section 3.3) ──────────────────────────────


def _sample_addition() -> str:
    n_digits = random.randint(1, 8)
    lo, hi = 10 ** (n_digits - 1), 10**n_digits - 1
    return f"{random.randint(lo, hi)} + {random.randint(lo, hi)}"


def _addition_teacher(x: str) -> str:
    return (
        "Solve the addition problem below by working right-to-left, "
        "one digit at a time, tracking carries in a scratchpad. "
        "After finishing, write your final numeric answer on its own line "
        "preceded by 'Final Answer:'.\n\n"
        "Example:\n"
        "Input: 247 + 189\n"
        "Scratchpad:\n"
        "  7 + 9 = 16  → write 6, carry 1\n"
        "  4 + 8 + 1 = 13 → write 3, carry 1\n"
        "  2 + 1 + 1 = 4  → write 4\n"
        "Final Answer: 436\n\n"
        f"Input: {x}\n"
        "Scratchpad:\n"
    )


def _addition_student(x: str) -> str:
    return f"Calculate: {x}\nAnswer:"


def _extract_addition(teacher_out: str) -> str:
    m = re.search(r"Final\s*Answer[:\s]*(\d+)", teacher_out)
    if m:
        return m.group(1).strip()
    # fallback: last contiguous number in output
    nums = re.findall(r"\d+", teacher_out)
    return nums[-1] if nums else ""


ADDITION_TASK = TaskDef(
    input_distribution=_sample_addition,
    teacher_template=_addition_teacher,
    student_template=_addition_student,
    answer_extractor=_extract_addition,
)


# ── 2b.  Sentiment classification (Section 3.1 style) ────────────────────────


def _sample_sentiment() -> str:
    """Placeholder – in production you'd sample from an unlabeled corpus or
    ask an LLM to synthesise reviews."""
    templates = [
        "The movie was absolutely wonderful and I loved every minute.",
        "Terrible film. Waste of two hours.",
        "It was okay, nothing special but not bad either.",
        "A masterpiece of modern cinema, truly inspiring.",
        "I couldn't stay awake. Boring from start to finish.",
        "Pretty decent, had a few good laughs.",
        "The acting was great but the plot was confusing.",
        "One of the worst films I have ever seen.",
        "Highly recommend this to anyone who appreciates good storytelling.",
        "Meh. It had its moments but overall forgettable.",
    ]
    return random.choice(templates)


def _sentiment_teacher(x: str) -> str:
    return (
        "Classify the following movie review as positive (1) or negative (0).\n"
        "Think step by step: first note the sentiment words, then decide.\n\n"
        'Example: "The film bored me to tears." → Step-by-step: "bored" is negative → 0\n\n'
        f'Review: "{x}"\n'
        "Step-by-step:"
    )


def _sentiment_student(x: str) -> str:
    return f'Review: "{x}"\nSentiment (0 or 1):'


def _extract_sentiment(teacher_out: str) -> str:
    # grab the last standalone 0 or 1
    matches = re.findall(r"\b([01])\b", teacher_out)
    return matches[-1] if matches else ""


SENTIMENT_TASK = TaskDef(
    input_distribution=_sample_sentiment,
    teacher_template=_sentiment_teacher,
    student_template=_sentiment_student,
    answer_extractor=_extract_sentiment,
)


BUILTIN_TASKS = {"addition": ADDITION_TASK, "sentiment": SENTIMENT_TASK}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TEACHER BACKENDS  –  pluggable generation sources
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    n_samples: int = 1  # completions per input (>1 → multiple distillation pairs)


class LocalTeacher:
    """Wraps any HuggingFace causal LM running locally (e.g. on RunPod)."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )

    @torch.no_grad()
    def generate(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outs = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_return_sequences=cfg.n_samples,
            do_sample=True,
        )
        n_prompt = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(o[n_prompt:], skip_special_tokens=True) for o in outs
        ]


class AnthropicTeacher:
    """Uses the Anthropic Messages API as the teacher."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        import anthropic  # lazy import – only needed when this backend is chosen

        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        results: list[str] = []
        for _ in range(cfg.n_samples):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=cfg.max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
            )
            results.append(resp.content[0].text)
        return results


def _make_teacher(teacher_type: str, teacher_model: str):
    if teacher_type == "anthropic":
        return AnthropicTeacher(model=teacher_model)
    return LocalTeacher(model_name=teacher_model)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET GENERATION  –  Phase 1 + 2 (sample → teacher → extract)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_distillation_dataset(
    task: TaskDef,
    teacher,
    n_examples: int = 1000,
    gen_cfg: GenerationConfig | None = None,
) -> list[dict]:
    """
    Returns a list of dicts, each containing:
        student_input  – what the student model sees at train/inference time
        target         – the extracted final answer it should learn to produce
        metadata       – raw_input + full teacher output (for debugging / analysis)
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()

    dataset: list[dict[str, Any]] = []
    failed = 0

    for i in range(n_examples):
        x = task.input_distribution()
        teacher_prompt = task.teacher_template(x)
        completions = teacher.generate(teacher_prompt, gen_cfg)

        for completion in completions:
            answer = task.answer_extractor(completion)
            if not answer:
                failed += 1
                continue
            dataset.append(
                {
                    "student_input": task.student_template(x),
                    "target": answer,
                    "metadata": {"raw_input": x, "teacher_output": completion},
                }
            )

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_examples}]  valid={len(dataset)}  failed_extractions={failed}")

    print(f"  Done.  {len(dataset)} valid pairs, {failed} failed extractions.")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  STUDENT FINE-TUNING  –  Phase 3  (LoRA on a causal LM)
# ═══════════════════════════════════════════════════════════════════════════════


class _DistillDataset(Dataset):
    """PyTorch Dataset that masks the prompt so the loss is only on target tokens."""

    def __init__(self, data: list[dict], tokenizer, max_len: int = 128):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        # We concatenate prompt + " " + target and mask the prompt in labels
        prompt = item["student_input"]
        target = item["target"]
        full_text = prompt + " " + target

        full_enc = self.tok(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tok(
            prompt + " ",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze()
        attn = full_enc["attention_mask"].squeeze()

        labels = input_ids.clone()
        # -100 on prompt tokens → no gradient signal there
        labels[: prompt_enc["input_ids"].shape[1]] = -100
        labels[attn == 0] = -100  # mask padding too

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


@dataclass
class FinetuneConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 8
    max_seq_len: int = 128
    grad_accum_steps: int = 1
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_dropout: float = 0.05
    # output
    adapter_save_path: str = "context_distillation_adapter"


def finetune_student(
    dataset: list[dict],
    cfg: FinetuneConfig | None = None,
) -> tuple:
    """
    LoRA-finetunes a causal LM on the distilled dataset.
    Returns (model, tokenizer) so you can do inference right away.
    """
    if cfg is None:
        cfg = FinetuneConfig()

    print(f"  Loading base model  {cfg.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # ── attach LoRA ──
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── data loader ──
    ds = _DistillDataset(dataset, tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # ── optimizer + scheduler ──
    total_steps = (len(loader) // cfg.grad_accum_steps) * cfg.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── training loop ──
    model.train()
    global_step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(model.device)
            attn = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            epoch_loss += out.loss.item()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % 50 == 0:
                print(
                    f"    Epoch {epoch+1} | step {step+1}/{len(loader)} "
                    f"| loss {epoch_loss/(step+1):.4f}"
                )

        print(f"  Epoch {epoch+1} done — avg loss {epoch_loss/len(loader):.4f}")

    # ── persist adapter ──
    model.save_pretrained(cfg.adapter_save_path)
    print(f"  Saved LoRA adapter → {cfg.adapter_save_path}/")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_student(
    model,
    tokenizer,
    task: TaskDef,
    n_eval: int = 200,
    max_new_tokens: int = 32,
) -> dict:
    """
    Samples inputs, runs the *student* prompt through the fine-tuned model,
    and compares the extracted output to the ground-truth answer.

    For addition we can compute ground truth directly; for other tasks you'd
    swap in your own scorer.
    """
    model.eval()
    correct = total = 0
    examples: list[dict] = []

    for _ in range(n_eval):
        x = task.input_distribution()
        prompt = task.student_template(x)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for eval
        )
        generated = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = task.answer_extractor(generated).strip()

        # ground truth (only works for addition – extend as needed)
        try:
            gt = str(eval(x))  # noqa: S307 – x is "A + B" from our own sampler
        except Exception:
            gt = None

        if gt is not None:
            total += 1
            if pred == gt:
                correct += 1
            examples.append(
                {"input": x, "pred": pred, "gt": gt, "correct": pred == gt}
            )

    acc = correct / total if total else 0.0
    print(f"  Eval: {correct}/{total} correct  ({acc:.1%})")
    return {"accuracy": acc, "examples": examples[:10]}


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TOP-LEVEL PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_pipeline(
    task: TaskDef = ADDITION_TASK,
    teacher_type: str = "anthropic",
    teacher_model: str = "claude-3-5-sonnet-latest",
    n_examples: int = 1000,
    finetune_cfg: FinetuneConfig | None = None,
    dataset_path: str = "distillation_dataset.json",
    load_existing_dataset: bool = False,
):
    """
    End-to-end context distillation.

    Set load_existing_dataset=True to skip generation and read from dataset_path.
    """
    print("=" * 62)
    print(" CONTEXT DISTILLATION PIPELINE")
    print("=" * 62)

    # ── Phase 1+2: dataset ─────────────────────────────────────────────────
    if load_existing_dataset and os.path.exists(dataset_path):
        print(f"\n[Phase 1]  Loading existing dataset from {dataset_path}")
        with open(dataset_path) as f:
            dataset = json.load(f)
        print(f"  → {len(dataset)} examples loaded")
    else:
        print(f"\n[Phase 1]  Generating dataset via {teacher_type} teacher …")
        teacher = _make_teacher(teacher_type, teacher_model)
        dataset = generate_distillation_dataset(
            task, teacher, n_examples=n_examples
        )
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"  → Saved {len(dataset)} pairs to {dataset_path}")

    # ── Phase 3: fine-tune ─────────────────────────────────────────────────
    if finetune_cfg is None:
        finetune_cfg = FinetuneConfig()

    print(f"\n[Phase 2]  Fine-tuning student  ({finetune_cfg.model_name}) …")
    model, tokenizer = finetune_student(dataset, finetune_cfg)

    # ── Phase 4: quick eval ────────────────────────────────────────────────
    print("\n[Phase 3]  Evaluating …")
    results = evaluate_student(model, tokenizer, task, n_eval=200)

    print("\n" + "=" * 62)
    print(f" Pipeline complete.  Accuracy: {results['accuracy']:.1%}")
    print("=" * 62)
    return model, tokenizer, results


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PROJECT-SPECIFIC: BUILD DATASET FROM EXISTING hint_and_rerun OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════


def build_dataset_from_hint_outputs(
    paths: list[str],
    *,
    require_improved_only: bool = True,
) -> list[dict]:
    """
    Build a distillation dataset directly from one or more *_with_hints.json files
    produced by hint_and_rerun.py.

    Each JSON file is expected to have the shape:
        {
          "results": [
            {
              "problem": ...,
              "rerun": {
                  "response": "... reasoning with behaviors ...",
                  "predicted": "...",   # extracted \\boxed{} answer
                  "correct": true/false,
                  ...
              },
              "delta": { "status": "improved" | "unchanged" | "still_wrong", ... },
              ...
            },
            ...
          ]
        }

    We treat the rerun-with-hints output as the *teacher* scratchpad y, and
    distil down to a student that sees only a minimal math prompt and learns
    to produce the final numeric answer.
    """
    from baseline_eval import extract_boxed_answer

    dataset: list[dict[str, Any]] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        items = blob.get("results", [])
        print(f"Loaded {len(items)} items from {path}")

        for item in items:
            problem = item.get("problem", "")
            rerun = item.get("rerun") or {}
            if not problem or not rerun:
                continue

            correct = bool(rerun.get("correct"))
            status = (item.get("delta") or {}).get("status")
            if require_improved_only:
                # keep only cases where hints actually helped
                if not (correct and status == "improved"):
                    continue
            else:
                if not correct:
                    continue

            teacher_response = rerun.get("response") or ""
            if not teacher_response:
                continue

            # final answer used as target
            target = rerun.get("predicted") or extract_boxed_answer(teacher_response)
            if not target:
                continue

            student_input = (
                "You are a helpful math assistant.\n"
                f"Problem: {problem}\n\n"
                "Solve the problem step by step. Put your final answer in \\boxed{}.\n"
                "Answer:"
            )

            dataset.append(
                {
                    "student_input": student_input,
                    "target": target,
                    "metadata": {
                        "problem": problem,
                        "teacher_response": teacher_response,
                        "status": status,
                        "source_file": os.path.abspath(path),
                    },
                }
            )

    print(f"Built distillation dataset with {len(dataset)} examples.")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Context distillation pipeline (Snell et al. 2022)"
    )

    parser.add_argument(
        "--phase",
        choices=["dataset", "finetune", "all"],
        default="dataset",
        help=(
            "dataset – generate & save only | "
            "finetune – load dataset & train | "
            "all – end-to-end"
        ),
    )
    parser.add_argument("--task", choices=list(BUILTIN_TASKS), default="addition")
    parser.add_argument(
        "--n", type=int, default=500, help="Number of distillation examples"
    )
    parser.add_argument(
        "--teacher", default="anthropic", choices=["anthropic", "local"]
    )
    parser.add_argument(
        "--teacher-model",
        default="claude-3-5-sonnet-latest",
        help="Anthropic or HF model id for the teacher.",
    )
    parser.add_argument(
        "--student-model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HF model id for the student.",
    )
    parser.add_argument(
        "--dataset-path",
        default="distillation_dataset.json",
        help="Where to save/load the distillation pairs.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    task = BUILTIN_TASKS[args.task]

    if args.phase == "dataset":
        # ── generate only ──
        print(
            f"Generating {args.n} examples for task={args.task} via {args.teacher} …"
        )
        teacher = _make_teacher(args.teacher, args.teacher_model)
        ds = generate_distillation_dataset(task, teacher, n_examples=args.n)
        with open(args.dataset_path, "w") as f:
            json.dump(ds, f, indent=2)
        print(f"Saved {len(ds)} examples → {args.dataset_path}")

    elif args.phase == "finetune":
        # ── load existing dataset & train ──
        print(f"Loading dataset from {args.dataset_path} …")
        with open(args.dataset_path) as f:
            ds = json.load(f)
        ft_cfg = FinetuneConfig(
            model_name=args.student_model,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model, tokenizer = finetune_student(ds, ft_cfg)
        evaluate_student(model, tokenizer, task)

    else:  # "all"
        ft_cfg = FinetuneConfig(
            model_name=args.student_model,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        run_pipeline(
            task=task,
            teacher_type=args.teacher,
            teacher_model=args.teacher_model,
            n_examples=args.n,
            finetune_cfg=ft_cfg,
            dataset_path=args.dataset_path,
        )

