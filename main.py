"""Chapter 4 experiment entry point.

Place this file at the project root. The implementation keeps the method defined
in Chapter 4 and exposes only experiment/scenario switches around it.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from getdata import GetDataSet
from src.models.model import choose_model
from src.utils.imbanlance import (
    dirichlet_split_noniid,
    split_imbalance_two,
    split_iid,
    split,
    split_NIID,
)
from src.utils.tool_utils import setup_seed

TRAINERS = {"hierarchical_server": "Hierarchical_server"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def model_payload_bits(model):
    """Return |theta| in bits for the complete transmitted ``state_dict``.

    Aggregation sends trainable parameters and model buffers, so communication
    energy must count the same payload.  The old code returned an MB number and
    used it as bits, underestimating transmission by millions of times.
    """
    return int(sum(tensor.numel() * tensor.element_size() * 8
                   for tensor in model.state_dict().values()))


def input_options(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Dynamic reputation and resource-aware HFL experiments",
    )
    # Reproducibility and task.
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id; use -1 for CPU")
    parser.add_argument("--dataset_name", choices=["mnist", "fashionmnist", "cifar10"], default="mnist")
    parser.add_argument("--model_name", default="mnist_cnn")
    parser.add_argument("--server", choices=TRAINERS, default="hierarchical_server")
    parser.add_argument("--round_num", type=int, default=100)
    parser.add_argument("--local_num", type=int, default=1, help="edge aggregations per cloud round")
    parser.add_argument("--local_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_of_clients", type=int, default=50)
    parser.add_argument("--num_of_MEC", type=int, default=5)
    parser.add_argument("--c_fraction", type=float, default=1.0)
    # Non-IID split: retain the original Chapter 3/4 settings.
    parser.add_argument("--method_division", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--dirichlet", type=float, default=0.3)
    parser.add_argument("--is_iid", type=str2bool, default=False)
    # Original system/energy parameters.
    parser.add_argument("--CPU_cycles", type=float, default=1e7)
    parser.add_argument("--coefficient", type=float, default=1e-28)
    parser.add_argument("--f_min", type=float, default=1e9)
    parser.add_argument("--f_max", type=float, default=4e9)
    parser.add_argument("--p_min", type=float, default=0.2)
    parser.add_argument("--p_max", type=float, default=0.5)
    parser.add_argument("--bandwidth", type=float, default=10e6)
    parser.add_argument("--noise_power", type=float, default=4e-21)
    parser.add_argument("--loss_exponent", type=float, default=3.0)
    parser.add_argument("--space", type=float, default=1000.0)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--initial_energy", type=float, default=25000.0)
    parser.add_argument("--cost_scale", type=float, default=1.0)
    parser.add_argument("--min_train_samples", type=int, default=1, help="minimum sample visits required to participate")
    parser.add_argument("--round_deadline", type=float, default=0.0, help="seconds; <=0 disables the time upper bound")

    # Chapter 4 reputation and pricing parameters.
    parser.add_argument("--initial_reputation", type=float, default=0.5)
    parser.add_argument("--beta0", type=float, default=0.9, help="historical coefficient in EMA")
    parser.add_argument("--attack_sensitivity", type=float, default=1.0)
    parser.add_argument("--sigma_epsilon", type=float, default=1e-12)
    parser.add_argument("--td_epsilon", type=float, default=1e-6)
    parser.add_argument("--td_max_iterations", type=int, default=100)
    parser.add_argument("--lambda0", type=float, default=500.0)
    parser.add_argument("--lambda_gamma", type=float, default=0.995)
    parser.add_argument("--lambda_min", type=float, default=1.0)
    parser.add_argument("--price_grid_step", type=float, default=5.0, help="cloud price grid spacing on [0, lambda_t]")
    parser.add_argument("--price_refinement_rounds", type=int, default=2)
    parser.add_argument("--price_refinement_points", type=int, default=101)
    parser.add_argument("--reputation_threshold", type=float, default=0.3)
    parser.add_argument("--target_accuracy", type=float, default=0.8)

    # Keep the aggregation/defence choice separate from the amount of data
    # clients are allowed to process.  ``auto`` preserves the original
    # behaviour of every experiment variant.
    parser.add_argument(
        "--contribution_policy",
        choices=["auto", "stackelberg", "fixed_budget", "full"],
        default="auto",
        help=(
            "client sample-allocation policy; fixed_budget is the fair-comparison "
            "mode and full is the unconstrained FedAvg upper-bound mode"
        ),
    )
    parser.add_argument(
        "--round_sample_budget",
        type=int,
        default=10000,
        help="system-wide sample-visit budget per global round in fixed_budget mode",
    )

    # Experiment variants. `proposed` is the complete Chapter 4 method.
    parser.add_argument(
        "--experiment_variant",
        choices=[
            "proposed", "fixed_ema", "edge_td_only", "cloud_td_only",
            "fixed_cost", "fixed_satisfaction", "fixed_price",
            "hierfavg", "trimmed_mean", "coordinate_median", "no_defense",
        ],
        default="proposed",
    )
    parser.add_argument("--trim_ratio", type=float, default=0.2)
    parser.add_argument(
        "--fixed_cloud_price", type=float, default=0.05,
        help="edge/client price used by the fixed_price baseline",
    )

    # Delayed/intermittent Byzantine scenarios.
    parser.add_argument("--client_is_byzantine", type=str2bool, default=True)
    parser.add_argument("--client_attack_method", choices=["fang", "ipm", "lie", "scaling", "mean", "full_mean"], default="mean")
    parser.add_argument("--n_client_byzantine", type=float, default=0.2, help="ratio (<1) or count")
    parser.add_argument("--edge_is_byzantine", type=str2bool, default=False)
    parser.add_argument("--edge_attack_method", choices=["fang", "ipm", "lie", "scaling", "mean", "full_mean"], default="mean")
    parser.add_argument("--n_edge_byzantine", type=float, default=0.2)
    parser.add_argument("--cloud_is_byzantine", type=str2bool, default=False, help="kept for compatibility; not a supported threat in Chapter 4")
    parser.add_argument("--attack_start_round", type=int, default=30, help="zero-based round; 30 means attack begins at displayed round 31")
    parser.add_argument("--attack_schedule", choices=["always", "delayed", "intermittent", "never"], default="delayed")
    parser.add_argument("--attack_on_rounds", type=int, default=5)
    parser.add_argument("--attack_off_rounds", type=int, default=5)

    # Outputs.
    parser.add_argument("--output_dir", default="results_chapter4")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--save_checkpoint", type=str2bool, default=True)
    parser.add_argument("--verbose_clients", type=str2bool, default=False)

    args = parser.parse_args(argv)
    options = vars(args)
    if not (0 < options["beta0"] < 1):
        parser.error("--beta0 must be in (0, 1)")
    if options["round_num"] <= 0 or options["num_of_clients"] <= 0 or options["num_of_MEC"] <= 0:
        parser.error("round/client/MEC counts must be positive")
    if options["num_of_clients"] < options["num_of_MEC"]:
        parser.error("num_of_clients must be at least num_of_MEC")
    if options["cloud_is_byzantine"]:
        parser.error("the Chapter 4 threat model assumes a trusted cloud; use edge attacks instead")
    if options["price_grid_step"] <= 0:
        parser.error("--price_grid_step must be positive")
    if options["price_refinement_rounds"] < 0 or options["price_refinement_points"] < 3:
        parser.error("price refinement requires non-negative rounds and at least 3 points")
    if options["round_sample_budget"] <= 0:
        parser.error("--round_sample_budget must be positive")

    setup_seed(options["seed"])
    compatible_models = {
        "mnist": {"mnist_cnn", "mnist_dnn"},
        "fashionmnist": {"fmnist_cnn"},
        "cifar10": {"cifar10_cnn", "cifar10_alexnet", "cifar10_vgg16"},
    }
    if options["model_name"].lower() not in compatible_models[options["dataset_name"]]:
        parser.error(
            f"model {options['model_name']} is incompatible with dataset {options['dataset_name']}; "
            f"choose one of {sorted(compatible_models[options['dataset_name']])}"
        )
    if options["dirichlet"] <= 0 or options["initial_energy"] <= 0:
        parser.error("dirichlet and initial_energy must be positive")
    if options["bandwidth"] <= 0 or options["CPU_cycles"] <= 0:
        parser.error("bandwidth and CPU_cycles must be positive")
    if options["theta"] != 1.0:
        parser.error("--theta is a legacy compatibility option and must be 1.0 for the Chapter 4 formula")
    dataset = GetDataSet(options["dataset_name"])
    # Compute communication payload from the selected instance; individual
    # model classes previously returned inconsistent MB values or no method.
    options["model_size"] = float(model_payload_bits(choose_model(options)))
    labels = dataset.trainLabel
    if options["method_division"] == 1:
        client_label, _ = dirichlet_split_noniid(
            labels, options["dirichlet"], options["num_of_clients"], seed=options["seed"]
        )
    elif options["method_division"] == 2:
        client_label = split_imbalance_two(labels, options["num_of_clients"], options["seed"])
    elif options["method_division"] == 3:
        client_label = split_iid(labels, options["num_of_clients"], options["seed"])
    elif options["method_division"] == 4:
        client_label = split(labels, options["num_of_clients"], options["seed"])
    else:
        client_label = split_NIID(labels, options["dirichlet"], options["num_of_clients"], options["seed"])
    if len(client_label) != options["num_of_clients"]:
        raise ValueError("data splitter did not return one index set per client")
    return client_label, options, dataset


def main(argv=None):
    client_label, options, dataset = input_options(argv)
    run_name = options["run_name"] or (
        f'{options["dataset_name"]}_{options["experiment_variant"]}_seed{options["seed"]}'
    )
    options["run_name"] = run_name
    run_dir = Path(options["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding="utf-8")

    module = importlib.import_module(f'src.fed_server.{options["server"]}')
    trainer = getattr(module, TRAINERS[options["server"]])(options, dataset, client_label)
    summary = trainer.train()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
