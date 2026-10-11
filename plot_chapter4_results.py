"""Generate Chapter 4 figures directly from the experiment CSV files.

Usage examples:
    python plot_chapter4_results.py --input_dir results_chapter4
    python plot_chapter4_results.py --input_dir results_chapter4 --attack_method mean

The script uses only the standard library plus matplotlib; pandas is not
required. Each subdirectory under ``input_dir`` is expected to contain the
``summary.json`` and CSV files produced by ``main.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover - depends on the local runtime
    raise SystemExit(
        "plot_chapter4_results.py requires matplotlib. Install it with "
        "'python -m pip install matplotlib' and run the command again."
    ) from error


def _number(value):
    if value is None or value == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def resolved_policy(summary, config):
    policy = summary.get("contribution_policy") or config.get("contribution_policy", "auto")
    if policy != "auto":
        return policy
    return "full" if summary.get("variant") in {"hierfavg", "no_defense"} else "stackelberg"


def _prepare_round_rows(rows):
    cumulative_samples = 0.0
    for row in rows:
        used = row.get("used_samples", float("nan"))
        if math.isfinite(used):
            cumulative_samples += used
        if not math.isfinite(row.get("cumulative_used_samples", float("nan"))):
            row["cumulative_used_samples"] = cumulative_samples
    return rows


def load_runs(input_dir, attack_method=None, variant=None, schedule=None, contribution_policy=None):
    runs = []
    for summary_path in sorted(Path(input_dir).glob("*/summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        if (
            attack_method
            and config.get("attack_schedule", "delayed") != "never"
            and config.get("client_attack_method", "mean") != attack_method
        ):
            continue
        if variant and summary.get("variant") != variant:
            continue
        if schedule and config.get("attack_schedule") != schedule:
            continue
        policy = resolved_policy(summary, config)
        if contribution_policy and policy != contribution_policy:
            continue
        runs.append({
            "dir": run_dir,
            "summary": summary,
            "config": config,
            "policy": policy,
            "budget": summary.get("round_sample_budget", config.get("round_sample_budget")),
            "round": _prepare_round_rows(read_csv(run_dir / "round_metrics.csv")) if (run_dir / "round_metrics.csv").exists() else [],
            "client": read_csv(run_dir / "client_metrics.csv") if (run_dir / "client_metrics.csv").exists() else [],
            "edge": read_csv(run_dir / "edge_metrics.csv") if (run_dir / "edge_metrics.csv").exists() else [],
        })
    return runs


def group_runs(runs):
    groups = defaultdict(list)
    policies = {run["policy"] for run in runs}
    schedules = {run["config"].get("attack_schedule", "delayed") for run in runs}
    attacks = {
        run["config"].get("client_attack_method", "mean") for run in runs
        if run["config"].get("attack_schedule", "delayed") != "never"
    }
    attacker_ratios = {run["config"].get("n_client_byzantine", 0.2) for run in runs}
    dirichlet_values = {run["config"].get("dirichlet", 0.2) for run in runs}
    initial_energies = {run["config"].get("initial_energy", 25000.0) for run in runs}
    budgets = {run["budget"] for run in runs if run["policy"] == "fixed_budget"}
    for run in runs:
        label = run["summary"].get("variant", run["dir"].name)
        if len(policies) > 1:
            label += f" | {run['policy']}"
        if run["policy"] == "fixed_budget" and len(budgets) > 1:
            label += f" q={run['budget']}"
        if len(schedules) > 1:
            label += f" | {run['config'].get('attack_schedule', 'delayed')}"
        if len(attacks) > 1 and run["config"].get("attack_schedule", "delayed") != "never":
            label += f" | {run['config'].get('client_attack_method', 'mean')}"
        if len(attacker_ratios) > 1:
            label += f" | f={run['config'].get('n_client_byzantine', 0.2)}"
        if len(dirichlet_values) > 1:
            label += f" | alpha={run['config'].get('dirichlet', 0.2)}"
        if len(initial_energies) > 1:
            label += f" | E={run['config'].get('initial_energy', 25000.0)}"
        groups[label].append(run)
    return dict(sorted(groups.items()))


def mean_std(values):
    valid = [float(value) for value in values if math.isfinite(float(value))]
    if not valid:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def aggregate_round_series(runs, table_name, field, extra_filter=None):
    by_round = defaultdict(list)
    for run in runs:
        for row in run[table_name]:
            if extra_filter and not extra_filter(row):
                continue
            value = row.get(field, float("nan"))
            if math.isfinite(value):
                by_round[int(row["round"])].append(value)
    x = sorted(by_round)
    y, error = zip(*(mean_std(by_round[round_id]) for round_id in x)) if x else ([], [])
    return np.asarray(x), np.asarray(y), np.asarray(error)


def plot_curve(ax, groups, table_name, field, title, ylabel, extra_filter=None):
    for label, runs in groups.items():
        x, y, error = aggregate_round_series(runs, table_name, field, extra_filter)
        if len(x) == 0:
            continue
        ax.plot(x, y, label=label, linewidth=1.8)
        ax.fill_between(x, y - error, y + error, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Global round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_efficiency_curve(ax, groups, x_field, title, xlabel):
    x_values = [
        row.get(x_field, float("nan"))
        for runs in groups.values() for run in runs for row in run["round"]
    ]
    x_values = [value for value in x_values if math.isfinite(value)]
    maximum = max(x_values, default=0.0)
    if maximum <= 0:
        ax.set_title(title)
        return
    grid = np.linspace(0.0, maximum, 200)
    for label, runs in groups.items():
        interpolated = []
        for run in runs:
            points = [
                (row.get(x_field, float("nan")), row.get("global_accuracy", float("nan")))
                for row in run["round"]
            ]
            points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
            if not points:
                continue
            x = np.asarray([0.0] + [point[0] for point in points])
            y = np.asarray([_number(run["summary"].get("initial_accuracy"))] + [point[1] for point in points])
            values = np.interp(grid, x, y)
            values[grid > x[-1]] = np.nan
            interpolated.append(values)
        if not interpolated:
            continue
        matrix = np.asarray(interpolated)
        valid = np.sum(np.isfinite(matrix), axis=0)
        y = np.divide(
            np.nansum(matrix, axis=0), valid,
            out=np.full(len(grid), np.nan), where=valid > 0,
        )
        error = np.zeros(len(grid))
        for index in range(len(grid)):
            values = matrix[:, index]
            values = values[np.isfinite(values)]
            error[index] = float(np.std(values)) if len(values) else np.nan
        mask = np.isfinite(y)
        ax.plot(grid[mask], y[mask], label=label, linewidth=1.8)
        ax.fill_between(grid[mask], y[mask] - error[mask], y[mask] + error[mask], alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _pairing_key(run):
    config, summary = run["config"], run["summary"]
    signature = tuple(
        config.get(field) for field in (
            "dataset_name", "model_name", "round_num", "local_num", "local_epoch",
            "num_of_clients", "num_of_MEC", "c_fraction", "method_division",
            "dirichlet", "initial_energy", "n_client_byzantine",
        )
    )
    return (
        summary.get("variant"), run["policy"], run["budget"],
        int(summary.get("seed", 0)), signature,
    )


def paired_drop_groups(runs, schedule=None):
    clean = {
        _pairing_key(run): run for run in runs
        if run["config"].get("attack_schedule", "delayed") == "never"
    }
    pairs = []
    for attack_run in runs:
        attack_schedule = attack_run["config"].get("attack_schedule", "delayed")
        if attack_schedule == "never" or (schedule and attack_schedule != schedule):
            continue
        clean_run = clean.get(_pairing_key(attack_run))
        if clean_run is None:
            continue
        clean_accuracy = {int(row["round"]): row["global_accuracy"] for row in clean_run["round"]}
        attack_accuracy = {int(row["round"]): row["global_accuracy"] for row in attack_run["round"]}
        start = 1 if attack_schedule == "always" else int(attack_run["config"].get("attack_start_round", 30)) + 1
        rounds = sorted(round_id for round_id in clean_accuracy if round_id in attack_accuracy and round_id >= start)
        if not rounds:
            continue
        drops = [clean_accuracy[round_id] - attack_accuracy[round_id] for round_id in rounds]
        pairs.append((attack_run, float(np.mean(drops)), max(max(drops), 0.0)))
    grouped = defaultdict(list)
    temporary_groups = group_runs([run for run, _, _ in pairs]) if pairs else {}
    labels_by_directory = {
        str(run["dir"]): label for label, group in temporary_groups.items() for run in group
    }
    for run, mean_drop, max_drop in pairs:
        grouped[labels_by_directory[str(run["dir"])]].append((mean_drop, max_drop))
    return dict(sorted(grouped.items()))


def save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(Path(output_dir) / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_values(groups, field):
    labels, means, errors = [], [], []
    for label, runs in groups.items():
        values = [_number(run["summary"].get(field)) for run in runs]
        values = [value for value in values if math.isfinite(value)]
        if values:
            labels.append(label)
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values)))
    return labels, np.asarray(means), np.asarray(errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot Chapter 4 experiment outputs")
    parser.add_argument("--input_dir", default="results_chapter4")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--attack_method", default=None, help="filter, e.g. mean or lie")
    parser.add_argument("--variant", default=None, help="plot one variant only")
    parser.add_argument("--schedule", default=None, help="filter attack schedule")
    parser.add_argument(
        "--contribution_policy",
        choices=["stackelberg", "fixed_budget", "full"],
        default=None,
        help="filter one sample-allocation policy",
    )
    args = parser.parse_args(argv)
    runs = load_runs(
        args.input_dir, args.attack_method, args.variant, args.schedule,
        args.contribution_policy,
    )
    if not runs:
        raise SystemExit("No matching summary.json files were found")
    output_dir = Path(args.output_dir or Path(args.input_dir) / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = group_runs(runs)
    first_config = runs[0]["config"]
    attack_start = int(first_config.get("attack_start_round", 30)) + 1

    # 1. Main convergence figure.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_curve(axes[0], groups, "round", "global_accuracy", "Global accuracy", "Accuracy")
    plot_curve(axes[1], groups, "round", "global_loss", "Global loss", "Cross-entropy loss")
    for ax in axes:
        ax.axvline(attack_start, color="black", linestyle="--", alpha=0.45, label="attack start")
        ax.legend(fontsize=8)
    save(fig, output_dir, "01_accuracy_loss.png")

    # 2. Client reputation: honest versus malicious.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, group in groups.items():
        x, malicious, error = aggregate_round_series(
            group, "client", "reputation_after", lambda row: row.get("malicious", 0) == 1
        )
        if len(x):
            ax.plot(x, malicious, label=f"{label} malicious")
        x, honest, error = aggregate_round_series(
            group, "client", "reputation_after", lambda row: row.get("malicious", 0) == 0
        )
        if len(x):
            ax.plot(x, honest, linestyle="--", label=f"{label} honest")
    ax.axvline(attack_start, color="black", linestyle="--", alpha=0.45)
    ax.set_title("Client reputation: honest versus malicious")
    ax.set_xlabel("Global round")
    ax.set_ylabel("EMA reputation")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save(fig, output_dir, "02_client_reputation.png")

    # 3. Adaptive anomaly response: sigma and beta.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_curve(axes[0], groups, "client", "sigma", "Client anomaly score", "sigma")
    plot_curve(axes[1], groups, "client", "beta", "Adaptive forgetting factor", "beta")
    for ax in axes:
        ax.axvline(attack_start, color="black", linestyle="--", alpha=0.45)
        ax.legend(fontsize=8)
    save(fig, output_dir, "03_sigma_beta.png")

    # 4. Components of Eq. (4.7).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, title in zip(
        axes,
        ["gradient_change", "loss_change", "reputation_deviation"],
        ["Gradient change", "Loss change", "Reputation deviation"],
    ):
        plot_curve(ax, groups, "client", field, title, field)
        ax.axvline(attack_start, color="black", linestyle="--", alpha=0.45)
        ax.legend(fontsize=8)
    save(fig, output_dir, "04_anomaly_components.png")

    # 5. Payment and detection quality.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_curve(axes[0], groups, "round", "malicious_payment_share", "Malicious payment share", "Share")
    plot_curve(axes[1], groups, "round", "malicious_detection_rate", "Detection and false-positive rates", "Rate")
    for label, group in groups.items():
        x, y, error = aggregate_round_series(group, "round", "low_reputation_honest_rate")
        if len(x):
            axes[1].plot(x, y, linestyle="--", label=f"{label} honest false positive")
    for ax in axes:
        ax.axvline(attack_start, color="black", linestyle="--", alpha=0.45)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    save(fig, output_dir, "05_payment_detection.png")

    # 6. Energy and effective contribution.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_curve(axes[0], groups, "round", "mean_remaining_energy", "Mean remaining energy", "Energy")
    plot_curve(axes[0], groups, "round", "min_remaining_energy", "Minimum remaining energy", "Energy")
    plot_curve(axes[1], groups, "round", "used_samples", "Effective sample visits", "Samples")
    plot_curve(axes[1], groups, "round", "energy_eligible_clients", "Energy-eligible clients", "Clients")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, output_dir, "06_energy_contribution.png")

    # 7. Price, cost, and satisfaction dynamics.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_curve(axes[0], groups, "round", "global_base_price", "Cloud base price R(t)", "Price")
    plot_curve(axes[1], groups, "client", "cost_per_unit", "Client marginal cost", "c_i(t)")
    plot_curve(axes[2], groups, "round", "satisfaction_lambda", "Satisfaction parameter", "lambda(t)")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, output_dir, "07_price_cost_satisfaction.png")

    # 8. Edge-level reputation for edge attack experiments.
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_curve(ax, groups, "edge", "reputation_after", "Edge reputation", "H_cloud,j(t)")
    plot_curve(ax, groups, "edge", "truth_model_change", "Edge truth-model change", "L2 change")
    ax.legend(fontsize=8)
    save(fig, output_dir, "08_edge_reputation.png")

    # 9. Summary comparison bars.
    metrics = [
        ("final_accuracy", "Final accuracy"),
        ("max_attack_accuracy_drop", "Max accuracy drop"),
        ("cumulative_energy", "Cumulative energy"),
        ("malicious_payment_share", "Malicious payment share"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (field, title) in zip(axes.flat, metrics):
        labels, values, errors = summary_values(groups, field)
        if labels:
            positions = np.arange(len(labels))
            ax.bar(positions, values, yerr=errors, capsize=3)
            ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    save(fig, output_dir, "09_summary_comparison.png")

    # 10. Accuracy-energy Pareto view.
    labels, accuracy, accuracy_error = summary_values(groups, "final_accuracy")
    _, energy, energy_error = summary_values(groups, "cumulative_energy")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if labels:
        ax.errorbar(energy, accuracy, xerr=energy_error, yerr=accuracy_error, fmt="o")
        for x_value, y_value, label in zip(energy, accuracy, labels):
            ax.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Accuracy-energy trade-off")
    ax.set_xlabel("Cumulative energy")
    ax.set_ylabel("Final accuracy")
    ax.grid(alpha=0.25)
    save(fig, output_dir, "10_accuracy_energy_pareto.png")

    # 11. Fair efficiency curves: the x-axis reflects actual consumed work.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_efficiency_curve(
        axes[0], groups, "cumulative_used_samples",
        "Accuracy versus cumulative sample visits", "Cumulative sample visits",
    )
    plot_efficiency_curve(
        axes[1], groups, "cumulative_energy",
        "Accuracy versus cumulative energy", "Cumulative energy",
    )
    save(fig, output_dir, "11_accuracy_samples_energy.png")

    # 12. Net attack degradation relative to each method's paired clean run.
    pair_runs = load_runs(
        args.input_dir, args.attack_method, args.variant, None,
        args.contribution_policy,
    )
    paired = paired_drop_groups(pair_runs, args.schedule)
    if paired:
        labels = list(paired)
        mean_drops = np.asarray([np.mean([row[0] for row in paired[label]]) for label in labels])
        max_drops = np.asarray([np.mean([row[1] for row in paired[label]]) for label in labels])
        positions = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5.5))
        ax.bar(positions - width / 2, mean_drops, width, label="Mean post-attack drop")
        ax.bar(positions + width / 2, max_drops, width, label="Maximum post-attack drop")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_ylabel("Clean accuracy - attacked accuracy")
        ax.set_title("Paired clean/attack degradation")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        save(fig, output_dir, "12_paired_attack_degradation.png")
    print(f"Generated {len(list(output_dir.glob('*.png')))} figures in {output_dir}")


if __name__ == "__main__":
    main()
