#!/usr/bin/env python3
"""
Build a context-distillation dataset from hint_and_rerun *_with_hints.json outputs,
then LoRA fine-tune the student (e.g. Qwen2.5-Math-1.5B) and save the adapter.

Usage
-----
# One or more hint output files (default: your results run)
  python run_hint_distillation.py --input results/run_20260203_064154/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json

# Optional: save dataset for inspection, change student model or adapter path
  python run_hint_distillation.py --input results/run_.../baseline_eval_*_with_hints.json \\
    --dataset-out math_hints_distill.json \\
    --student-model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --adapter-path context_distillation_adapter \\
    --epochs 3 --batch-size 4 --max-seq-len 512
"""

import argparse
import json
import os

from context_distillation import (
    build_dataset_from_hint_outputs,
    FinetuneConfig,
    finetune_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build distillation dataset from hint outputs and train LoRA adapter."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path(s) to *_with_hints.json from hint_and_rerun.py. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-out",
        default=None,
        help="If set, save the built dataset JSON here before training.",
    )
    parser.add_argument(
        "--student-model",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="HuggingFace model to fine-tune (student).",
    )
    parser.add_argument(
        "--adapter-path",
        default="context_distillation_adapter",
        help="Directory where the LoRA adapter will be saved.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for training (math problems need 512+).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--require-improved-only",
        action="store_true",
        default=True,
        help="Use only examples where hints helped (correct + status=improved).",
    )
    parser.add_argument(
        "--no-require-improved-only",
        action="store_false",
        dest="require_improved_only",
        help="Include all correct reruns, not only improved.",
    )
    parser.add_argument(
        "--metacognitive-only",
        action="store_true",
        default=True,
        help="Use only files from metacognitive approach (behavior handbooks); skip mock_hints / use_solution_hint runs.",
    )
    parser.add_argument(
        "--no-metacognitive-only",
        action="store_false",
        dest="metacognitive_only",
        help="Include all hint files regardless of mock_hints / use_solution_hint.",
    )
    args = parser.parse_args()

    paths = [os.path.abspath(p) for p in args.input]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Input file not found: {p}")

    print("Building distillation dataset from hint outputs …")
    dataset = build_dataset_from_hint_outputs(
        paths,
        require_improved_only=args.require_improved_only,
        metacognitive_only=args.metacognitive_only,
    )
    if not dataset:
        raise SystemExit(
            "No examples in dataset. Check that input JSONs have 'results' with "
            "'rerun.correct' and 'delta.status' (e.g. 'improved')."
        )

    if args.dataset_out:
        with open(args.dataset_out, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} examples → {args.dataset_out}")

    print("Fine-tuning student (LoRA) …")
    cfg = FinetuneConfig(
        model_name=args.student_model,
        adapter_save_path=os.path.abspath(args.adapter_path),
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    model, tokenizer = finetune_student(dataset, cfg)
    print(f"Done. Adapter saved to: {cfg.adapter_save_path}")
    print(f"Run distilled eval with: --adapter-path {args.adapter_path}")


if __name__ == "__main__":
    main()
