"""Run a reproducible Chapter 4 experiment matrix.

Example:
    python run_chapter4_experiments.py --seeds 30 31 32 --variants proposed no_defense \
        --contribution_policy fixed_budget --schedules never delayed
Extra arguments after ``--`` are forwarded to main.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[30])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "proposed", "no_defense", "hierfavg", "trimmed_mean",
            "coordinate_median", "fixed_ema", "edge_td_only",
            "cloud_td_only", "fixed_cost", "fixed_satisfaction", "fixed_price",
        ],
    )
    parser.add_argument("--attack_methods", nargs="+", default=["mean"])
    parser.add_argument("--output_dir", default="results_chapter4")
    parser.add_argument("--dataset_name", default="mnist")
    parser.add_argument("--model_name", default="mnist_cnn")
    parser.add_argument(
        "--contribution_policy",
        choices=["auto", "stackelberg", "fixed_budget", "full"],
        default="auto",
    )
    parser.add_argument("--round_sample_budget", type=int, default=10000)
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=["always", "delayed", "intermittent", "never"],
        default=None,
        help="optional clean/attack matrix; use 'never delayed' for paired runs",
    )
    parser.add_argument("--dry_run", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    controlled = {
        "--seed", "--experiment_variant", "--client_attack_method",
        "--dataset_name", "--model_name", "--contribution_policy",
        "--round_sample_budget", "--output_dir", "--run_name",
    }
    duplicates = sorted(controlled.intersection(forwarded))
    if duplicates:
        parser.error(
            f"pass {', '.join(duplicates)} before '--' using the batch-runner options"
        )
    if args.schedules and "--attack_schedule" in forwarded:
        parser.error("do not forward --attack_schedule when --schedules is used")

    schedules = args.schedules or [None]
    for schedule in schedules:
        # A clean run contains no attack, so one copy can be paired with every
        # attack family for the same method, policy, budget, and seed.
        schedule_attacks = args.attack_methods[:1] if schedule == "never" else args.attack_methods
        for attack in schedule_attacks:
            for variant in args.variants:
                for seed in args.seeds:
                    name_parts = [variant]
                    if args.contribution_policy != "auto":
                        name_parts.extend([args.contribution_policy, f"q{args.round_sample_budget}"])
                    if schedule == "never":
                        name_parts.append("clean")
                    else:
                        name_parts.append(attack)
                    if schedule is not None and schedule != "never":
                        name_parts.append(schedule)
                    name_parts.append(f"seed{seed}")
                    run_name = "_".join(name_parts)
                    command = [
                        sys.executable,
                        "main.py",
                        "--seed", str(seed),
                        "--experiment_variant", variant,
                        "--client_attack_method", attack,
                        "--dataset_name", args.dataset_name,
                        "--model_name", args.model_name,
                        "--contribution_policy", args.contribution_policy,
                        "--round_sample_budget", str(args.round_sample_budget),
                        "--output_dir", args.output_dir,
                        "--run_name", run_name,
                    ]
                    if schedule is not None:
                        command.extend(["--attack_schedule", schedule])
                    command.extend(forwarded)
                    print(" ".join(command), flush=True)
                    if not args.dry_run:
                        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

