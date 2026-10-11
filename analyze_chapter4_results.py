"""Aggregate Chapter 4 runs and pair clean/attacked experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SUMMARY_METRICS = [
    "final_accuracy",
    "best_accuracy",
    "last_5_round_accuracy",
    "max_attack_accuracy_drop",
    "cumulative_used_samples",
    "cumulative_energy",
    "cumulative_client_payment",
    "malicious_payment_share",
    "post_attack_malicious_payment_share",
    "detection_delay",
    "recovery_delay",
    "final_mean_remaining_energy",
    "final_reputation_gap",
    "final_malicious_detection_rate",
    "final_honest_false_positive_rate",
]


def resolved_policy(summary, config):
    policy = summary.get("contribution_policy") or config.get("contribution_policy", "auto")
    if policy != "auto":
        return policy
    return "full" if summary.get("variant") in {"hierfavg", "no_defense"} else "stackelberg"


def read_rounds(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                "round": int(row["round"]),
                "global_accuracy": float(row["global_accuracy"]),
            }
            for row in csv.DictReader(handle)
        ]


def load_runs(input_dir):
    runs = []
    for summary_path in sorted(Path(input_dir).glob("*/summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        policy = resolved_policy(summary, config)
        budget = summary.get("round_sample_budget")
        if budget is None and policy == "fixed_budget":
            budget = config.get("round_sample_budget", 10000)
        runs.append({
            "dir": run_dir,
            "summary": summary,
            "config": config,
            "policy": policy,
            "budget": budget,
            "rounds": read_rounds(run_dir / "round_metrics.csv"),
        })
    return runs


def metric_stats(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    if not values:
        return "", ""
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def aggregate_summaries(runs):
    groups = defaultdict(list)
    for run in runs:
        summary, config = run["summary"], run["config"]
        key = (
            summary["variant"],
            run["policy"],
            run["budget"] if run["budget"] is not None else "",
            "none" if config.get("attack_schedule", "delayed") == "never"
            else config.get("client_attack_method", "mean"),
            config.get("attack_schedule", "delayed"),
            config.get("attack_start_round", 30),
            config.get("n_client_byzantine", 0.2),
            config.get("dirichlet", 0.2),
            config.get("initial_energy", 25000.0),
        )
        groups[key].append(summary)

    records = []
    for key, rows in sorted(groups.items(), key=str):
        (
            variant, policy, budget, attack, schedule, attack_start,
            attacker_ratio, dirichlet, initial_energy,
        ) = key
        record = {
            "variant": variant,
            "contribution_policy": policy,
            "round_sample_budget": budget,
            "attack_method": attack,
            "attack_schedule": schedule,
            "attack_start_round": attack_start,
            "n_client_byzantine": attacker_ratio,
            "dirichlet": dirichlet,
            "initial_energy": initial_energy,
            "runs": len(rows),
        }
        for metric in SUMMARY_METRICS:
            metric_mean, metric_std = metric_stats(rows, metric)
            record[f"{metric}_mean"] = metric_mean
            record[f"{metric}_std"] = metric_std
        records.append(record)
    return records


def pairing_key(run):
    config, summary = run["config"], run["summary"]
    # Every field below must be identical for the clean and attacked run.
    experiment_signature = tuple(
        config.get(field) for field in (
            "dataset_name", "model_name", "round_num", "local_num", "local_epoch",
            "num_of_clients", "num_of_MEC", "c_fraction", "method_division",
            "dirichlet", "initial_energy", "n_client_byzantine",
        )
    )
    return (
        summary["variant"],
        run["policy"],
        run["budget"],
        int(summary.get("seed", config.get("seed", 0))),
        experiment_signature,
    )


def paired_robustness(runs):
    clean = {
        pairing_key(run): run
        for run in runs
        if run["config"].get("attack_schedule", "delayed") == "never"
    }
    pairs = []
    for attack_run in runs:
        schedule = attack_run["config"].get("attack_schedule", "delayed")
        if schedule == "never":
            continue
        clean_run = clean.get(pairing_key(attack_run))
        if clean_run is None:
            continue
        clean_by_round = {row["round"]: row["global_accuracy"] for row in clean_run["rounds"]}
        attack_by_round = {row["round"]: row["global_accuracy"] for row in attack_run["rounds"]}
        start = 1 if schedule == "always" else int(attack_run["config"].get("attack_start_round", 30)) + 1
        shared = sorted(
            round_id for round_id in clean_by_round
            if round_id in attack_by_round and round_id >= start
        )
        if not shared:
            continue
        drops = [clean_by_round[round_id] - attack_by_round[round_id] for round_id in shared]
        pairs.append({
            "variant": attack_run["summary"]["variant"],
            "contribution_policy": attack_run["policy"],
            "round_sample_budget": attack_run["budget"] if attack_run["budget"] is not None else "",
            "attack_method": attack_run["config"].get("client_attack_method", "mean"),
            "attack_schedule": schedule,
            "attack_start_round": int(attack_run["config"].get("attack_start_round", 30)),
            "n_client_byzantine": attack_run["config"].get("n_client_byzantine", 0.2),
            "dirichlet": attack_run["config"].get("dirichlet", 0.2),
            "initial_energy": attack_run["config"].get("initial_energy", 25000.0),
            "seed": int(attack_run["summary"].get("seed", 0)),
            "paired_rounds": len(shared),
            "clean_final_accuracy": clean_by_round[shared[-1]],
            "attacked_final_accuracy": attack_by_round[shared[-1]],
            "final_accuracy_drop": clean_by_round[shared[-1]] - attack_by_round[shared[-1]],
            "mean_post_attack_accuracy_drop": mean(drops),
            "max_post_attack_accuracy_drop": max(max(drops), 0.0),
        })
    return pairs


def aggregate_pairs(pairs):
    groups = defaultdict(list)
    fields = (
        "variant", "contribution_policy", "round_sample_budget",
        "attack_method", "attack_schedule", "attack_start_round",
        "n_client_byzantine", "dirichlet", "initial_energy",
    )
    for row in pairs:
        groups[tuple(row[field] for field in fields)].append(row)
    metrics = [
        "clean_final_accuracy", "attacked_final_accuracy", "final_accuracy_drop",
        "mean_post_attack_accuracy_drop", "max_post_attack_accuracy_drop",
    ]
    records = []
    for key, rows in sorted(groups.items(), key=str):
        record = dict(zip(fields, key))
        record["paired_runs"] = len(rows)
        for metric in metrics:
            metric_mean, metric_std = metric_stats(rows, metric)
            record[f"{metric}_mean"] = metric_mean
            record[f"{metric}_std"] = metric_std
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results_chapter4")
    parser.add_argument("--output", default=None)
    parser.add_argument("--paired_output", default=None)
    parser.add_argument("--paired_runs_output", default=None)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output) if args.output else input_dir / "aggregate_summary.csv"
    paired_output = Path(args.paired_output) if args.paired_output else input_dir / "paired_robustness_summary.csv"
    paired_runs_output = (
        Path(args.paired_runs_output)
        if args.paired_runs_output else input_dir / "paired_robustness_runs.csv"
    )

    runs = load_runs(input_dir)
    if not runs:
        raise SystemExit(f"No summary.json files were found under {input_dir}")
    write_csv(output, aggregate_summaries(runs))
    pairs = paired_robustness(runs)
    write_csv(paired_runs_output, pairs)
    write_csv(paired_output, aggregate_pairs(pairs))
    print(output)
    if pairs:
        print(paired_runs_output)
        print(paired_output)
    else:
        print("No matched clean/attack pairs were found; paired CSV files were not written.")


if __name__ == "__main__":
    main()
