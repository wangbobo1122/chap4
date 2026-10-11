"""Three-layer reputation-adaptive incentive mechanism used in Chapter 4.

The implementation intentionally exposes ablation switches around the existing
method instead of changing the method itself.  Byzantine attacks are applied to
model *updates* before truth discovery, so reputation and aggregation observe
the same transmitted object.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.attacks.byzantine_attacks import get_malicious_updates
from src.fed_MEC.MEC_server import MEC_server, weighted_state_dict
from src.fed_client.Client_server import Client_server
from src.models.model import choose_model
from src.optimizers.gd import GD


def _cpu_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _float_keys(state):
    return [key for key in sorted(state) if torch.is_floating_point(state[key])]


def _flatten_delta(state, reference):
    keys = _float_keys(reference)
    if not keys:
        return torch.empty(0)
    return torch.cat([(state[key].detach().cpu().float() - reference[key].detach().cpu().float()).reshape(-1) for key in keys])


def _state_from_delta(delta, reference):
    result = _cpu_state(reference)
    position = 0
    for key in _float_keys(reference):
        size = reference[key].numel()
        block = delta[position : position + size].reshape(reference[key].shape)
        result[key] = (reference[key].detach().cpu().float() + block).to(reference[key].dtype)
        position += size
    if position != delta.numel():
        raise ValueError("the flattened update does not match the reference model")
    return result


def _positive_median(values, epsilon):
    finite = [float(v) for v in values if np.isfinite(v)]
    return max(float(np.median(finite)), epsilon) if finite else 1.0


class Hierarchical_server:
    def __init__(self, options, dataset, clients_label):
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.rng = np.random.default_rng(int(options.get("seed", 0)))
        self.py_rng = random.Random(int(options.get("seed", 0)))

        self.model = choose_model(options)
        self.optimizer = GD(self.model.parameters(), lr=options["lr"])
        self.latest_global_model = self.get_model_parameters()
        self.MEC_servers = self.get_MEC(int(options["num_of_MEC"]))
        self.all_clients = self.get_clients(dataset, clients_label, self.MEC_servers)
        self.selected_MEC = list(self.MEC_servers)
        self.B = float(options["bandwidth"])

        self.td_epsilon = float(options.get("td_epsilon", 1e-6))
        self.td_max_iterations = int(options.get("td_max_iterations", 100))
        self.global_base_price = 0.0
        self.satisfaction_param = float(options.get("lambda0", 500.0))
        self.initial_acc = 0.0
        self.best_acc = 0.0

        variant = options.get("experiment_variant", "proposed")
        self.variant = variant
        self.use_client_td = variant not in {
            "cloud_td_only", "hierfavg", "trimmed_mean", "coordinate_median", "no_defense"
        }
        self.use_cloud_td = variant not in {
            "edge_td_only", "hierfavg", "trimmed_mean", "coordinate_median", "no_defense"
        }
        self.use_adaptive_ema = variant not in {"fixed_ema", "no_defense"}
        self.use_dynamic_cost = variant not in {"fixed_cost", "no_defense"}
        self.use_dynamic_satisfaction = variant not in {"fixed_satisfaction", "no_defense"}
        requested_policy = options.get("contribution_policy", "auto")
        if requested_policy == "auto":
            requested_policy = "full" if variant in {"hierfavg", "no_defense"} else "stackelberg"
        self.contribution_policy = requested_policy
        self.use_stackelberg = self.contribution_policy == "stackelberg"
        self.use_trimmed_mean = variant == "trimmed_mean"
        self.use_coordinate_median = variant == "coordinate_median"

        self.client_attacker_ids = self._choose_attackers(
            len(self.all_clients), options.get("n_client_byzantine", 0), salt=17
        ) if options.get("client_is_byzantine", False) else set()
        self.edge_attacker_ids = self._choose_attackers(
            len(self.MEC_servers), options.get("n_edge_byzantine", 0), salt=19
        ) if options.get("edge_is_byzantine", False) else set()
        for client in self.all_clients:
            client.is_byzantine = client.id in self.client_attacker_ids
        for edge in self.MEC_servers:
            edge.is_byzantine = edge.id in self.edge_attacker_ids

        run_name = options.get("run_name") or f"run_seed{options.get('seed', 0)}"
        self.run_dir = Path(options.get("output_dir", "results_chapter4")) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.round_records = []
        self.client_records = []
        self.edge_records = []

    # Construction ------------------------------------------------------------------
    def get_model_parameters(self):
        return _cpu_state(self.model.state_dict())

    def set_model_parameters(self, model_parameters):
        current = self.model.state_dict()
        state = {key: model_parameters[key].to(dtype=value.dtype, device=value.device) for key, value in current.items()}
        self.model.load_state_dict(state, strict=True)
        self.latest_global_model = self.get_model_parameters()

    def get_MEC(self, number):
        servers = []
        center = float(self.options["space"]) / 2.0
        radius = center * math.sqrt(2) / 2
        bandwidths = self.rng.dirichlet(np.ones(number)) * float(self.options["bandwidth"])
        for i in range(number):
            angle = i * 2 * math.pi / number
            location = (center + radius * math.cos(angle), center + radius * math.sin(angle))
            model = copy.deepcopy(self.model).cpu()
            optimizer = GD(model.parameters(), lr=self.options["lr"])
            servers.append(
                MEC_server(self.options, model, i, optimizer, self.dataset, self.clients_label, location, bandwidths[i])
            )
        return servers

    def get_clients(self, dataset, clients_label, MEC_servers):
        labels = copy.deepcopy(list(clients_label))
        self.py_rng.shuffle(labels)
        clients = []
        for i in range(int(self.options["num_of_clients"])):
            model = copy.deepcopy(self.model).cpu()
            optimizer = GD(model.parameters(), lr=self.options["lr"])
            indices = np.asarray(labels[i], dtype=np.int64)
            local_dataset = TensorDataset(
                torch.as_tensor(dataset.trainData[indices]),
                torch.as_tensor(dataset.trainLabel[indices]).long(),
            )
            client = Client_server(self.options, i, model, optimizer, local_dataset, 0.0, 0.0)
            clients.append(client)
        self.assign_clients_to_MEC(clients, MEC_servers)
        return clients

    def assign_clients_to_MEC(self, clients, MEC_servers):
        shuffled = list(clients)
        self.py_rng.shuffle(shuffled)
        groups = np.array_split(np.asarray(shuffled, dtype=object), len(MEC_servers))
        for edge, group in zip(MEC_servers, groups):
            edge.selected_clients = list(group)
            for client in edge.selected_clients:
                client.MEC = edge
                angle = self.rng.uniform(0, 2 * math.pi)
                radius = self.rng.uniform(1.0, 50.0)
                client.x = edge.x + radius * math.cos(angle)
                client.y = edge.y + radius * math.sin(angle)

    def _choose_attackers(self, total, amount, salt):
        amount = float(amount)
        count = int(math.ceil(total * amount)) if 0 < amount < 1 else int(amount)
        count = min(max(count, 0), total)
        rng = np.random.default_rng(int(self.options.get("seed", 0)) + salt)
        return set(map(int, rng.choice(total, size=count, replace=False))) if count else set()

    # Scenario and participant selection --------------------------------------------
    def attack_active(self, round_index):
        schedule = self.options.get("attack_schedule", "delayed")
        if schedule == "never":
            return False
        if schedule == "always":
            return True
        start = int(self.options.get("attack_start_round", 30))
        if round_index < start:
            return False
        if schedule == "delayed":
            return True
        on = max(int(self.options.get("attack_on_rounds", 5)), 1)
        off = max(int(self.options.get("attack_off_rounds", 5)), 0)
        return (round_index - start) % (on + off) < on

    def _select_participants(self, edge, round_index):
        clients = edge.selected_clients
        fraction = min(max(float(self.options.get("c_fraction", 1.0)), 0.0), 1.0)
        count = min(len(clients), max(1, int(math.ceil(len(clients) * fraction))))
        rng = np.random.default_rng(int(self.options.get("seed", 0)) + 7919 * (round_index + 1) + edge.id)
        indices = rng.choice(len(clients), size=count, replace=False)
        edge.active_clients = [clients[int(index)] for index in sorted(indices)]
        return edge.active_clients

    # Chapter 4 truth discovery and attacks -----------------------------------------
    def _truth_discovery(self, models, reference):
        if len(models) == 1:
            return [1.0], _cpu_state(models[0]), 1
        updates = torch.stack([_flatten_delta(model, reference) for model in models])
        truth = updates.mean(dim=0)
        epsilon = self.td_epsilon
        weights = torch.ones(len(models), dtype=torch.float32)
        iterations = 0
        for iterations in range(1, self.td_max_iterations + 1):
            distances = torch.linalg.vector_norm(updates - truth, dim=1)
            total = distances.sum() + epsilon * len(models)
            weights = torch.log(total / (distances + epsilon)).clamp_min(0.0)
            if not torch.isfinite(weights).all() or float(weights.sum()) <= epsilon:
                weights = torch.ones_like(weights)
            new_truth = (updates * weights[:, None]).sum(dim=0) / weights.sum()
            # PDF Eq. (4.16) uses an absolute L2 convergence threshold.
            absolute_change = float(torch.linalg.vector_norm(new_truth - truth))
            if absolute_change < epsilon:
                truth = new_truth
                break
            truth = new_truth
        # Recompute weights using the final truth rather than the previous
        # iteration's truth.  Max-normalisation yields the EMA input in [0, 1].
        distances = torch.linalg.vector_norm(updates - truth, dim=1)
        total = distances.sum() + epsilon * len(models)
        weights = torch.log(total / (distances + epsilon)).clamp_min(0.0)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= epsilon:
            weights = torch.ones_like(weights)
        normalized = (weights / weights.max().clamp_min(epsilon)).cpu().tolist()
        return normalized, _state_from_delta(truth, reference), iterations

    def _trimmed_aggregate(self, models, reference):
        updates = torch.stack([_flatten_delta(model, reference) for model in models])
        trim = int(math.floor(float(self.options.get("trim_ratio", 0.2)) * len(models)))
        trim = min(trim, max((len(models) - 1) // 2, 0))
        if trim:
            sorted_updates = torch.sort(updates, dim=0).values
            aggregate = sorted_updates[trim : len(models) - trim].mean(dim=0)
        else:
            aggregate = updates.mean(dim=0)
        return _state_from_delta(aggregate, reference)

    def _coordinate_median_aggregate(self, models, reference):
        """Coordinate-wise median baseline over model updates."""
        updates = torch.stack([_flatten_delta(model, reference) for model in models])
        aggregate = torch.median(updates, dim=0).values
        return _state_from_delta(aggregate, reference)

    def _make_malicious_vectors(self, benign, count, method):
        """Generate attack vectors and expose any controlled small-group fallback."""
        try:
            malicious = get_malicious_updates(
                benign,
                benign.mean(dim=0),
                count,
                attack_method=method,
            ).detach().cpu().float()
            if malicious.ndim == 1:
                malicious = malicious.unsqueeze(0).repeat(count, 1)
            if len(malicious) != count or not torch.isfinite(malicious).all():
                raise ValueError("attack returned an invalid tensor")
        except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as error:
            # Fang/IPM need enough benign samples to estimate a statistic.  A
            # small-group fallback is explicit and logged; all other failures
            # must surface instead of silently changing the experiment.
            if method not in {"fang", "ipm"} or len(benign) >= 3:
                raise RuntimeError(f"attack {method} failed") from error
            center = benign.mean(dim=0)
            malicious = (-10.0 * center).unsqueeze(0).repeat(count, 1)
            self.last_attack_fallback = True
        return malicious

    def _inject_attacks(self, models, reference, malicious_positions, method):
        self.last_attack_fallback = False
        if not malicious_positions:
            return models
        attacked = [_cpu_state(model) for model in models]
        all_updates = torch.stack([_flatten_delta(model, reference) for model in models])
        benign_positions = [i for i in range(len(models)) if i not in malicious_positions]
        benign = all_updates[benign_positions] if benign_positions else all_updates
        malicious = self._make_malicious_vectors(benign, len(malicious_positions), method)
        for row, position in enumerate(malicious_positions):
            attacked[position] = _state_from_delta(malicious[row], reference)
        return attacked

    def _inject_gradient_attacks(self, actors, malicious_positions, method):
        """Apply the configured attack to the gradient signal used by Eq. (4.4)."""
        if not malicious_positions:
            return
        gradients = torch.stack([actor.current_gradient_vector for actor in actors])
        benign_positions = [i for i in range(len(actors)) if i not in malicious_positions]
        benign = gradients[benign_positions] if benign_positions else gradients
        malicious = self._make_malicious_vectors(benign, len(malicious_positions), method)
        for row, position in enumerate(malicious_positions):
            actor = actors[position]
            old = actor.current_gradient_vector
            actor.round_gradient_sum += (malicious[row] - old) * float(actor.used_samples)
            actor.current_gradient_vector = malicious[row].clone()

    def _ema_reputation_update(self, actors, instantaneous, components, available):
        """Apply PDF Eqs. (4.7)-(4.11) once at the end of a global round."""
        eps = float(self.options.get("sigma_epsilon", 1e-12))
        medians = []
        for values, present in zip(components, available):
            observed = [float(value) for value, ok in zip(values, present) if ok and np.isfinite(value)]
            medians.append(_positive_median(observed, eps))
        beta0 = float(self.options.get("beta0", 0.9))
        sensitivity = float(self.options.get("attack_sensitivity", 1.0))
        old_values = []
        for i, actor in enumerate(actors):
            old = float(actor.history_reputation)
            sigma = sum(
                float(values[i]) / median
                for values, median, component_present in zip(components, medians, available)
                if component_present[i]
            )
            beta = beta0 * math.exp(-sensitivity * max(sigma, 0.0)) if self.use_adaptive_ema else beta0
            beta = min(max(beta, 0.0), beta0)
            actor.instant_weight = float(np.clip(instantaneous[i], 0.0, 1.0))
            actor.last_sigma = float(sigma)
            actor.last_beta = float(beta)
            old_values.append(old)
            actor.history_reputation = float(np.clip(beta * old + (1.0 - beta) * actor.instant_weight, 0.0, 1.0))
        return old_values

    def _update_client_reputations(self, actors, instantaneous):
        """Update client H_i(t) from real local gradients and round loss."""
        gradient_changes, loss_changes, reputation_deviations = [], [], []
        gradient_available, loss_available, rep_available = [], [], []
        for actor, weight in zip(actors, instantaneous):
            current_gradient = actor.round_gradient_sum / max(actor.round_gradient_weight, 1) if actor.round_gradient_sum is not None else None
            previous_gradient = getattr(actor, "previous_gradient_vector", None)
            has_gradient = current_gradient is not None and previous_gradient is not None
            gradient_changes.append(float(torch.linalg.vector_norm(current_gradient - previous_gradient)) if has_gradient else 0.0)
            gradient_available.append(has_gradient)
            current_loss = actor.round_loss_sum / max(actor.round_loss_weight, 1) if actor.round_loss_weight else None
            has_loss = current_loss is not None and actor.previous_loss is not None
            loss_changes.append(abs(float(current_loss) - float(actor.previous_loss)) if has_loss else 0.0)
            loss_available.append(has_loss)
            reputation_deviations.append(abs(float(actor.history_reputation) - float(weight)))
            rep_available.append(True)
            actor.last_gradient_change = gradient_changes[-1]
            actor.last_loss_change = loss_changes[-1]
            actor.last_reputation_deviation = reputation_deviations[-1]
            if current_gradient is not None:
                actor._round_average_gradient = current_gradient.detach().cpu().clone()
            actor._round_average_loss = current_loss
        old_values = self._ema_reputation_update(
            actors, instantaneous,
            [gradient_changes, loss_changes, reputation_deviations],
            [gradient_available, loss_available, rep_available],
        )
        for actor in actors:
            if getattr(actor, "_round_average_gradient", None) is not None:
                actor.previous_gradient_vector = actor._round_average_gradient.clone()
            if getattr(actor, "_round_average_loss", None) is not None:
                actor.previous_loss = actor._round_average_loss
        return old_values

    def _update_edge_reputations(self, actors, truth_models, instantaneous):
        """Update edge H_cloud,j using consecutive edge truth models."""
        model_changes, rep_deviations = [], []
        model_available, rep_available = [], []
        for actor, model, weight in zip(actors, truth_models, instantaneous):
            previous = getattr(actor, "previous_truth_model", None)
            available = previous is not None
            model_changes.append(float(torch.linalg.vector_norm(_flatten_delta(model, previous))) if available else 0.0)
            model_available.append(available)
            rep_deviations.append(abs(float(actor.history_reputation) - float(weight)))
            rep_available.append(True)
            actor.last_model_change = model_changes[-1]
            actor.last_reputation_deviation = rep_deviations[-1]
        old_values = self._ema_reputation_update(
            actors, instantaneous, [model_changes, rep_deviations, [0.0] * len(actors)],
            [model_available, rep_available, [False] * len(actors)],
        )
        for actor, model in zip(actors, truth_models):
            actor.previous_truth_model = _cpu_state(model)
        return old_values

    # Stackelberg stages -------------------------------------------------------------
    def _satisfaction(self, round_index):
        initial = float(self.options.get("lambda0", 500.0))
        if self.use_dynamic_satisfaction:
            value = initial * float(self.options.get("lambda_gamma", 0.995)) ** round_index
            return max(value, float(self.options.get("lambda_min", 1.0)))
        return initial

    @staticmethod
    def _edge_response(edge, R):
        """Solve PDF Eqs. (4.42)-(4.44) over all KKT partitions."""
        viable = [
            client for client in edge.active_clients
            if client.max_samples_by_energy > 0 and client.history_reputation > 0 and client.cost_per_unit > 0
        ]
        value = max(float(R) * edge.history_reputation, 0.0)
        if not viable or value <= 0:
            return 0.0, {client.id: 0.0 for client in edge.active_clients}, 0.0, 0.0
        thresholds = sorted(
            set(
                [0.0, value]
                + [
                    min(max(client.cost_per_unit * client.max_samples_by_energy / client.history_reputation, 0.0), value)
                    for client in viable
                ]
            )
        )
        candidates = set(thresholds)
        for lower, upper in zip(thresholds[:-1], thresholds[1:]):
            midpoint = (lower + upper) / 2.0
            saturated = [c for c in viable if c.cost_per_unit * c.max_samples_by_energy / c.history_reputation <= midpoint]
            interior = [
                c for c in viable
                if c not in saturated
            ]
            reputation_cost = sum(c.history_reputation / c.cost_per_unit for c in interior)
            reputation_squared_cost = sum(c.history_reputation**2 / c.cost_per_unit for c in interior)
            saturated_samples = sum(c.max_samples_by_energy for c in saturated)
            if reputation_cost > 0:
                stationary = (
                    value * reputation_squared_cost - saturated_samples
                ) / (2.0 * reputation_cost)
                candidates.add(min(max(stationary, lower), upper))

        best_price = best_utility = best_weighted = 0.0
        best_q = {client.id: 0.0 for client in edge.active_clients}
        for price in sorted(candidates):
            quantities = {
                client.id: (
                    min(float(client.max_samples_by_energy), price * client.history_reputation / client.cost_per_unit)
                )
                for client in viable
            }
            weighted = sum(c.history_reputation * quantities[c.id] for c in viable)
            total_samples = sum(quantities.values())
            # Edge revenue contains reputation; payments to clients do not.
            utility = value * weighted - price * total_samples
            if utility > best_utility + 1e-12:
                best_price, best_utility, best_weighted, best_q = price, utility, weighted, quantities
        return best_price, best_q, best_weighted, best_utility

    def _constrained_cloud_price(self, initial_price):
        """Active-set evaluation of the generalized KKT root in PDF Eq. (4.52)."""
        price = max(float(initial_price), 0.0)
        previous_partition = None
        for _ in range(20):
            A = B = 0.0
            partition = []
            for edge in self.MEC_servers:
                _, quantities, _, _ = self._edge_response(edge, price)
                viable = [
                    client for client in edge.active_clients
                    if client.max_samples_by_energy > 0
                    and client.history_reputation > 0
                    and client.cost_per_unit > 0
                ]
                saturated = [
                    client for client in viable
                    if quantities.get(client.id, 0.0) >= client.max_samples_by_energy - 1e-9
                ]
                interior = [client for client in viable if client not in saturated]
                partition.append(tuple(client.id for client in saturated))
                linear_sum = sum(c.history_reputation / c.cost_per_unit for c in interior)
                squared_sum = sum(c.history_reputation ** 2 / c.cost_per_unit for c in interior)
                if linear_sum <= 0:
                    continue
                alpha = edge.history_reputation * squared_sum / (2.0 * linear_sum)
                delta = sum(c.max_samples_by_energy for c in saturated) / (2.0 * linear_sum)
                A += edge.history_reputation * alpha * squared_sum
                B += edge.history_reputation * (
                    sum(c.history_reputation * c.max_samples_by_energy for c in saturated)
                    - delta * squared_sum
                )
            if A <= 0:
                return 0.0
            quadratic_b = 2.0 * A * (1.0 + B) + A * B
            quadratic_c = B * (1.0 + B) - self.satisfaction_param * A
            discriminant = quadratic_b ** 2 - 8.0 * A ** 2 * quadratic_c
            if discriminant < 0:
                return price
            candidate = (-quadratic_b + math.sqrt(discriminant)) / (4.0 * A ** 2)
            candidate = min(max(candidate, 0.0), self.satisfaction_param)
            current_partition = tuple(partition)
            if current_partition == previous_partition and abs(candidate - price) < 1e-9:
                return candidate
            previous_partition = current_partition
            price = candidate
        return price

    def _stage1_cloud_price(self, active_clients, round_index):
        self.satisfaction_param = self._satisfaction(round_index)
        if self.contribution_policy == "fixed_budget":
            self.global_base_price = 0.0
            self._planned_prices = {edge.id: 0.0 for edge in self.MEC_servers}
            self._planned_quantities = self._allocate_fixed_budget(active_clients)
            self.price_grid_final_step = 0.0
            return 0.0
        if self.contribution_policy == "full":
            self.global_base_price = 0.0
            self._planned_prices = {edge.id: 0.0 for edge in self.MEC_servers}
            self._planned_quantities = {
                client.id: float(client.max_samples_by_energy) for client in active_clients
            }
            self.price_grid_final_step = 0.0
            return 0.0
        if self.variant == "fixed_price":
            fixed_price = max(float(self.options.get("fixed_cloud_price", 0.05)), 0.0)
            self.global_base_price = fixed_price
            self._planned_prices = {edge.id: fixed_price for edge in self.MEC_servers}
            self._planned_quantities = {}
            for edge in self.MEC_servers:
                _, quantities, _, _ = self._edge_response(edge, fixed_price)
                self._planned_quantities.update(quantities)
            self.price_grid_final_step = 0.0
            return fixed_price
        best = (float("-inf"), 0.0, {}, {})
        def evaluate_grid(grid):
            nonlocal best
            for R in grid:
                prices, quantities = {}, {}
                S = 0.0
                for edge in self.MEC_servers:
                    price, q, weighted, _ = self._edge_response(edge, float(R))
                    prices[edge.id] = price
                    quantities.update(q)
                    S += edge.history_reputation * weighted
                utility = self.satisfaction_param * math.log1p(max(S, 0.0)) - float(R) * S
                if utility > best[0] + 1e-12:
                    best = (utility, float(R), prices, quantities)

        # PDF Eq. (4.47): in the unconstrained interior, Sigma_total(R)=Theta*R
        # and R* has a closed form.  Evaluate it explicitly, then retain the
            # finite-candidate search for clients hitting Q_max.
        theta = 0.0
        for edge in self.MEC_servers:
            squared_sum = sum(
                client.history_reputation ** 2 / client.cost_per_unit
                for client in edge.active_clients
                if client.max_samples_by_energy > 0 and client.cost_per_unit > 0
            )
            linear_sum = sum(
                client.history_reputation / client.cost_per_unit
                for client in edge.active_clients
                if client.max_samples_by_energy > 0 and client.cost_per_unit > 0
            )
            if linear_sum > 0:
                theta += (
                    edge.history_reputation ** 2 * squared_sum ** 2
                    / (2.0 * linear_sum)
                )
        closed_form = 0.0
        if theta > 0:
            closed_form = (math.sqrt(1.0 + 2.0 * theta * self.satisfaction_param) - 1.0) / (2.0 * theta)
            closed_form = min(max(closed_form, 0.0), self.satisfaction_param)
        constrained_closed_form = self._constrained_cloud_price(closed_form)
        evaluate_grid([closed_form, constrained_closed_form])
        step = float(self.options.get("price_grid_step", 5.0))
        evaluate_grid(list(np.arange(0.0, self.satisfaction_param, step)) + [self.satisfaction_param])
        points = int(self.options.get("price_refinement_points", 101))
        for _ in range(int(self.options.get("price_refinement_rounds", 2))):
            center = best[1]
            lower, upper = max(0.0, center - step), min(self.satisfaction_param, center + step)
            grid = np.linspace(lower, upper, points)
            evaluate_grid(grid)
            step = (upper - lower) / max(points - 1, 1)
        self.price_grid_final_step = step
        _, self.global_base_price, self._planned_prices, self._planned_quantities = best
        return self.global_base_price

    def _stage2_edge_price(self, edge):
        if not self.use_stackelberg:
            edge.price_to_clients = 0.0
            return 0.0
        edge.price_to_clients = float(self._planned_prices.get(edge.id, 0.0))
        return edge.price_to_clients

    def _allocate_fixed_budget(self, active_clients):
        """Allocate one neutral round budget in proportion to local data size.

        The allocation is independent of the aggregation method and reputation,
        which makes FedAvg, robust aggregation, and TD directly comparable.  It
        remains subject to each client's hard energy limit.
        """
        quantities = {client.id: 0 for client in active_clients}
        if not active_clients:
            return quantities
        caps = {
            client.id: max(int(client.max_samples_by_energy), 0)
            for client in active_clients
        }
        target = min(
            int(self.options.get("round_sample_budget", 10000)),
            sum(caps.values()),
        )
        total_data = sum(max(int(client.data_size), 0) for client in active_clients)
        if target <= 0 or total_data <= 0:
            return quantities

        raw = {
            client.id: target * float(client.data_size) / float(total_data)
            for client in active_clients
        }
        for client in active_clients:
            quantities[client.id] = min(caps[client.id], int(math.floor(raw[client.id])))

        # Largest-remainder allocation preserves the requested system budget.
        # Repeating the deterministic order also redistributes quota from a
        # client that has already reached its energy cap.
        remaining = target - sum(quantities.values())
        order = sorted(
            active_clients,
            key=lambda client: (-(raw[client.id] - math.floor(raw[client.id])), client.id),
        )
        while remaining > 0:
            progressed = False
            for client in order:
                if quantities[client.id] >= caps[client.id]:
                    continue
                quantities[client.id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break
        return {client_id: float(value) for client_id, value in quantities.items()}

    def _stage3_client_contribution(self, client, local_step):
        total = int(math.floor(self._planned_quantities.get(client.id, 0.0)))
        steps = max(int(self.options.get("local_num", 1)), 1)
        base, remainder = divmod(total, steps)
        requested = base + int(local_step < remainder)
        client.round_planned_samples = total
        return client.plan_contribution(requested, remaining_uploads=steps - local_step)

    # Evaluation and output ----------------------------------------------------------
    @property
    def device(self):
        gpu = int(self.options.get("gpu", -1))
        return torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")

    def evaluate_global(self):
        self.model.to(self.device).eval()
        data = TensorDataset(torch.as_tensor(self.dataset.testData), torch.as_tensor(self.dataset.testLabel).long())
        loader = DataLoader(data, batch_size=max(128, int(self.options.get("batch_size", 32))), shuffle=False)
        total_loss = total_correct = total_seen = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                batch = labels.size(0)
                total_loss += float(F.cross_entropy(logits, labels).item()) * batch
                total_correct += int(logits.argmax(1).eq(labels).sum().item())
                total_seen += batch
        self.model.cpu()
        return total_correct / max(total_seen, 1), total_loss / max(total_seen, 1)

    @staticmethod
    def _write_csv(path, records):
        if not records:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    def _save_results(self, summary):
        self._write_csv(self.run_dir / "round_metrics.csv", self.round_records)
        self._write_csv(self.run_dir / "client_metrics.csv", self.client_records)
        self._write_csv(self.run_dir / "edge_metrics.csv", self.edge_records)
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.options.get("save_checkpoint", True):
            torch.save(self.get_model_parameters(), self.run_dir / "global_model.pt")

    # Training -----------------------------------------------------------------------
    def train(self):
        """Run the complete Cloud-Edge-Client process for every global round."""
        # Step 0: evaluate the randomly initialised global model as the baseline.
        initial_accuracy, initial_loss = self.evaluate_global()
        self.initial_acc = initial_accuracy
        self.best_acc = initial_accuracy
        cumulative_energy = cumulative_payment = cumulative_malicious_payment = 0.0
        cumulative_used_samples = 0
        post_attack_payment = post_attack_malicious_payment = 0.0
        detection_round = None
        progress = tqdm(range(int(self.options["round_num"])), desc="Chapter 4 HFL")

        for round_index in progress:
            # Step 1: freeze H(t-1), select participants, and update energy cost c_i(t).
            started = time.perf_counter()
            displayed_round = round_index + 1
            active_attack = self.attack_active(round_index)
            self.round_attack_fallback = False
            global_reference = self.get_model_parameters()
            active_clients = []
            reputation_before = {client.id: client.history_reputation for client in self.all_clients}

            for edge in self.MEC_servers:
                active_clients.extend(self._select_participants(edge, round_index))
            for client in active_clients:
                client.energy_before_round = client.current_energy
                client.update_dynamic_cost(fixed=not self.use_dynamic_cost)
                client.round_used_samples = 0
                client.round_compute_energy = 0.0
                client.round_communication_energy = 0.0
                client.round_total_energy = 0.0
                client.round_payment = 0.0
                client.round_utility = 0.0
                client.round_gradient_sum = None
                client.round_gradient_weight = 0
                client.round_loss_sum = 0.0
                client.round_loss_weight = 0

            # Step 2: backward induction: cloud R*, edge p_j*, client q_i*.
            self._stage1_cloud_price(active_clients, round_index)
            edge_models = []
            edge_td_iterations = []
            round_client_energy = round_client_payment = round_malicious_payment = 0.0
            round_used_samples = 0
            round_client_actor_by_id = {}
            round_client_score_by_id = {}

            for edge in self.MEC_servers:
                # Step 3: broadcast the cloud model, train clients, attack uploads,
                # and run edge-side TD to obtain w_i(t) and Theta_truth,j(t).
                edge.set_model_parameters(global_reference)
                edge.round_weighted_contribution = 0.0
                edge.round_used_samples = 0
                edge.edge_revenue = edge.client_payment = edge.edge_utility = 0.0
                self._stage2_edge_price(edge)
                for local_step in range(int(self.options.get("local_num", 1))):
                    reference = edge.get_model_parameters()
                    transmitted_models = []
                    sample_counts = []
                    losses = []
                    for client in edge.active_clients:
                        client.set_parameters(reference)
                        client.global_base_price = self.global_base_price
                        client.price_from_edge = edge.price_to_clients
                        self._stage3_client_contribution(client, local_step)
                        local_model = client.local_train(round_index, local_step)
                        transmitted_models.append(local_model)
                        sample_counts.append(client.used_samples)
                        losses.append(client.last_train_loss if client.last_train_loss is not None else 0.0)

                    malicious_positions = [
                        i for i, client in enumerate(edge.active_clients)
                        if active_attack and client.id in self.client_attacker_ids and client.used_samples > 0
                    ]
                    transmitted_models = self._inject_attacks(
                        transmitted_models, reference, malicious_positions, self.options.get("client_attack_method", "mean")
                    )
                    self.round_attack_fallback = self.round_attack_fallback or self.last_attack_fallback

                    eligible = [i for i, count in enumerate(sample_counts) if count > 0]
                    if eligible:
                        eligible_models = [transmitted_models[i] for i in eligible]
                        eligible_clients = [edge.active_clients[i] for i in eligible]
                        eligible_losses = [losses[i] for i in eligible]
                        malicious_eligible_positions = [
                            position for position, original in enumerate(eligible)
                            if original in malicious_positions
                        ]
                        self._inject_gradient_attacks(
                            eligible_clients,
                            malicious_eligible_positions,
                            self.options.get("client_attack_method", "mean"),
                        )
                        self.round_attack_fallback = self.round_attack_fallback or self.last_attack_fallback
                        if self.use_client_td:
                            scores, aggregate, td_iterations = self._truth_discovery(eligible_models, reference)
                            # Reputation is committed once per global round;
                            # retain the final local-step TD observation here.
                            for actor, score in zip(eligible_clients, scores):
                                round_client_actor_by_id[actor.id] = actor
                                round_client_score_by_id[actor.id] = score
                        elif self.use_trimmed_mean:
                            scores = [1.0] * len(eligible)
                            aggregate = self._trimmed_aggregate(eligible_models, reference)
                            td_iterations = 0
                        elif self.use_coordinate_median:
                            scores = [1.0] * len(eligible)
                            aggregate = self._coordinate_median_aggregate(eligible_models, reference)
                            td_iterations = 0
                        else:
                            scores = [1.0] * len(eligible)
                            aggregate = weighted_state_dict(eligible_models, [sample_counts[i] for i in eligible])
                            td_iterations = 0
                        edge.set_model_parameters(aggregate)
                        edge_td_iterations.append(td_iterations)
                    else:
                        edge_td_iterations.append(0)

                    for client in edge.active_clients:
                        energy = client.settle_energy()
                        payment = client.price_from_edge * reputation_before[client.id] * client.used_samples
                        utility = payment - 0.5 * client.cost_per_unit * client.used_samples**2
                        client.last_payment = payment
                        client.last_utility = utility
                        client.round_used_samples += client.used_samples
                        client.round_compute_energy += client.last_compute_energy
                        client.round_communication_energy += client.last_communication_energy
                        client.round_total_energy += energy
                        client.round_payment += payment
                        client.round_utility += utility
                        edge.round_used_samples += client.used_samples
                        edge.round_weighted_contribution += reputation_before[client.id] * client.used_samples
                        # PDF Eq. (4.25) defines the edge's expenditure as p_j*q_i;
                        # the client's own utility follows Eq. (4.22), p_j*H_i*q_i.
                        edge.client_payment += client.price_from_edge * client.used_samples
                        round_used_samples += client.used_samples
                        round_client_energy += energy
                        round_client_payment += payment
                        if client.id in self.client_attacker_ids:
                            round_malicious_payment += payment

                edge.edge_revenue = (
                    self.global_base_price * edge.history_reputation * edge.round_weighted_contribution
                )
                edge.edge_utility = edge.edge_revenue - edge.client_payment
                edge.share_from_cloud = edge.edge_revenue
                edge_models.append(edge.get_model_parameters())

            # Step 4: use system-wide medians and commit each client reputation
            # exactly once, after every edge has finished its local steps.
            if self.use_client_td and round_client_actor_by_id:
                actors = list(round_client_actor_by_id.values())
                self._update_client_reputations(
                    actors, [round_client_score_by_id[actor.id] for actor in actors]
                )

            # Step 5: attack edge uploads if configured, then run cloud-side TD
            # to obtain W_j(t), the global truth, and H_cloud,j(t).
            edge_old_reputations = [edge.history_reputation for edge in self.MEC_servers]
            eligible_edge_positions = [
                i for i, edge in enumerate(self.MEC_servers) if edge.round_used_samples > 0
            ]
            eligible_edges = [self.MEC_servers[i] for i in eligible_edge_positions]
            eligible_edge_models = [edge_models[i] for i in eligible_edge_positions]
            malicious_edge_positions = [
                i for i, edge in enumerate(eligible_edges)
                if active_attack and edge.id in self.edge_attacker_ids
            ]
            eligible_edge_models = self._inject_attacks(
                eligible_edge_models,
                global_reference,
                malicious_edge_positions,
                self.options.get("edge_attack_method", "mean"),
            ) if eligible_edge_models else []
            self.round_attack_fallback = self.round_attack_fallback or self.last_attack_fallback
            if not eligible_edge_models:
                cloud_aggregate = global_reference
                cloud_td_iterations = 0
            elif self.use_cloud_td:
                edge_scores, cloud_aggregate, cloud_td_iterations = self._truth_discovery(
                    eligible_edge_models, global_reference
                )
                if eligible_edges:
                    updated_old = self._update_edge_reputations(
                        eligible_edges,
                        # The cloud evaluates the model it actually receives;
                        # this includes any edge-level Byzantine manipulation.
                        eligible_edge_models,
                        edge_scores,
                    )
                    for edge, old in zip(eligible_edges, updated_old):
                        edge_old_reputations[edge.id] = old
            elif self.use_trimmed_mean:
                cloud_aggregate = self._trimmed_aggregate(eligible_edge_models, global_reference)
                cloud_td_iterations = 0
            elif self.use_coordinate_median:
                cloud_aggregate = self._coordinate_median_aggregate(eligible_edge_models, global_reference)
                cloud_td_iterations = 0
            else:
                cloud_aggregate = weighted_state_dict(
                    eligible_edge_models, [edge.round_used_samples for edge in eligible_edges]
                )
                cloud_td_iterations = 0
            self.set_model_parameters(cloud_aggregate)

            # Step 6: evaluate the new global model and persist all observables.
            accuracy, loss = self.evaluate_global()
            self.best_acc = max(self.best_acc, accuracy)
            total_effective = sum(
                edge_old_reputations[i] * edge.round_weighted_contribution
                for i, edge in enumerate(self.MEC_servers)
            )
            cloud_payment = self.global_base_price * total_effective
            cloud_utility = self.satisfaction_param * math.log1p(max(total_effective, 0.0)) - cloud_payment
            cumulative_energy += round_client_energy
            cumulative_used_samples += round_used_samples
            cumulative_payment += round_client_payment
            cumulative_malicious_payment += round_malicious_payment
            if active_attack:
                post_attack_payment += round_client_payment
                post_attack_malicious_payment += round_malicious_payment

            malicious_reps = [c.history_reputation for c in self.all_clients if c.id in self.client_attacker_ids]
            honest_reps = [c.history_reputation for c in self.all_clients if c.id not in self.client_attacker_ids]
            threshold = float(self.options.get("reputation_threshold", 0.3))
            if (
                detection_round is None and active_attack and malicious_reps
                and float(np.mean(malicious_reps)) <= threshold
            ):
                detection_round = displayed_round

            max_violation = max(
                [
                    max(0.0, client.round_total_energy - client.energy_before_round)
                    for client in active_clients
                ]
                or [0.0]
            )
            record = {
                "round": displayed_round,
                "global_accuracy": accuracy,
                "global_loss": loss,
                "attack_active": int(active_attack),
                "attack_fallback": int(self.round_attack_fallback),
                "global_base_price": self.global_base_price,
                "price_grid_final_step": getattr(self, "price_grid_final_step", 0.0),
                "satisfaction_lambda": self.satisfaction_param,
                "active_clients": len(active_clients),
                "energy_eligible_clients": sum(
                    client.max_samples_by_energy >= int(self.options.get("min_train_samples", 1))
                    for client in active_clients
                ),
                "used_samples": round_used_samples,
                "planned_samples": int(sum(self._planned_quantities.values())),
                "round_sample_budget": (
                    int(self.options.get("round_sample_budget", 10000))
                    if self.contribution_policy == "fixed_budget" else ""
                ),
                "cumulative_used_samples": cumulative_used_samples,
                "round_energy": round_client_energy,
                "cumulative_energy": cumulative_energy,
                "mean_remaining_energy": float(np.mean([c.current_energy for c in self.all_clients])),
                "min_remaining_energy": float(np.min([c.current_energy for c in self.all_clients])),
                "client_payment": round_client_payment,
                "cumulative_client_payment": cumulative_payment,
                "malicious_payment": round_malicious_payment,
                "malicious_payment_share": round_malicious_payment / max(round_client_payment, 1e-12),
                "cloud_payment": cloud_payment,
                "cloud_utility": cloud_utility,
                "mean_client_reputation": float(np.mean([c.history_reputation for c in self.all_clients])),
                "mean_honest_reputation": float(np.mean(honest_reps)) if honest_reps else float("nan"),
                "mean_malicious_reputation": float(np.mean(malicious_reps)) if malicious_reps else float("nan"),
                "malicious_detection_rate": float(np.mean(np.asarray(malicious_reps) <= threshold)) if malicious_reps else 0.0,
                "low_reputation_honest_rate": float(np.mean(np.asarray(honest_reps) <= threshold)) if honest_reps else 0.0,
                "mean_client_td_iterations": float(np.mean(edge_td_iterations)) if edge_td_iterations else 0.0,
                "cloud_td_iterations": cloud_td_iterations,
                "max_energy_constraint_violation": max_violation,
                "runtime_seconds": time.perf_counter() - started,
            }
            self.round_records.append(record)

            selected_ids = {client.id for client in active_clients}
            for client in self.all_clients:
                stats = client.last_train_stats if client.id in selected_ids else {}
                self.client_records.append({
                    "round": displayed_round,
                    "client_id": client.id,
                    "edge_id": client.MEC.id,
                    "selected": int(client.id in selected_ids),
                    "malicious": int(client.id in self.client_attacker_ids),
                    "attack_active": int(active_attack and client.id in self.client_attacker_ids),
                    "attack_applied": int(
                        active_attack and client.id in self.client_attacker_ids
                        and client.id in selected_ids and getattr(client, "round_used_samples", 0) > 0
                    ),
                    "reputation_before": reputation_before[client.id],
                    "reputation_after": client.history_reputation,
                    "instant_weight": client.instant_weight,
                    "beta": client.last_beta,
                    "sigma": client.last_sigma,
                    "gradient_change": getattr(client, "last_gradient_change", ""),
                    "loss_change": getattr(client, "last_loss_change", ""),
                    "reputation_deviation": getattr(client, "last_reputation_deviation", ""),
                    "price": client.price_from_edge if client.id in selected_ids else 0.0,
                    "planned_samples": getattr(client, "round_planned_samples", 0) if client.id in selected_ids else 0,
                    "used_samples": getattr(client, "round_used_samples", 0) if client.id in selected_ids else 0,
                    "max_samples_by_energy": client.max_samples_by_energy,
                    "contribution_ratio": (
                        getattr(client, "round_used_samples", 0) / max(client.max_data_visits, 1)
                        if client.id in selected_ids else 0.0
                    ),
                    "cost_per_unit": client.cost_per_unit,
                    "compute_energy": getattr(client, "round_compute_energy", 0.0) if client.id in selected_ids else 0.0,
                    "communication_energy": getattr(client, "round_communication_energy", 0.0) if client.id in selected_ids else 0.0,
                    "remaining_energy": client.current_energy,
                    "payment": getattr(client, "round_payment", 0.0) if client.id in selected_ids else 0.0,
                    "utility": getattr(client, "round_utility", 0.0) if client.id in selected_ids else 0.0,
                    "train_loss": stats.get("loss", ""),
                    "train_accuracy": stats.get("acc", ""),
                })
            for i, edge in enumerate(self.MEC_servers):
                self.edge_records.append({
                    "round": displayed_round,
                    "edge_id": edge.id,
                    "malicious": int(edge.id in self.edge_attacker_ids),
                    "attack_active": int(active_attack and edge.id in self.edge_attacker_ids),
                    "attack_applied": int(
                        active_attack and edge.id in self.edge_attacker_ids and edge.round_used_samples > 0
                    ),
                    "reputation_before": edge_old_reputations[i],
                    "reputation_after": edge.history_reputation,
                    "instant_weight": edge.instant_weight,
                    "beta": edge.last_beta,
                    "sigma": edge.last_sigma,
                    "truth_model_change": getattr(edge, "last_model_change", ""),
                    "reputation_deviation": getattr(edge, "last_reputation_deviation", ""),
                    "price_to_clients": edge.price_to_clients,
                    "used_samples": edge.round_used_samples,
                    "weighted_contribution": edge.round_weighted_contribution,
                    "revenue": edge.edge_revenue,
                    "client_expenditure": edge.client_payment,
                    "utility": edge.edge_utility,
                })
            progress.set_postfix(acc=f"{accuracy:.4f}", loss=f"{loss:.4f}", R=f"{self.global_base_price:.3g}")

        tail = self.round_records[-min(5, len(self.round_records)) :]
        schedule = self.options.get("attack_schedule", "delayed")
        start_display = 1 if schedule == "always" else int(self.options.get("attack_start_round", 30)) + 1
        pre_attack = [row["global_accuracy"] for row in self.round_records if row["round"] < start_display]
        attack_rows = [row for row in self.round_records if row["attack_active"]]
        pre_attack_accuracy = float(np.mean(pre_attack)) if pre_attack else initial_accuracy
        attack_min_accuracy = min([row["global_accuracy"] for row in attack_rows], default=None)
        recovery_threshold = 0.95 * pre_attack_accuracy
        if attack_rows:
            minimum_round = next(
                row["round"] for row in attack_rows
                if row["global_accuracy"] == attack_min_accuracy
            )
            recovery_round = next(
                (
                    row["round"] for row in self.round_records
                    if row["round"] > minimum_round and row["global_accuracy"] >= recovery_threshold
                ),
                None,
            )
        else:
            recovery_round = None
        target = float(self.options.get("target_accuracy", 0.8))
        time_to_target = next(
            (row["round"] for row in self.round_records if row["global_accuracy"] >= target), None
        )
        summary = {
            "variant": self.variant,
            "contribution_policy": self.contribution_policy,
            "round_sample_budget": (
                int(self.options.get("round_sample_budget", 10000))
                if self.contribution_policy == "fixed_budget" else None
            ),
            "seed": int(self.options.get("seed", 0)),
            "client_attacker_ids": sorted(self.client_attacker_ids),
            "edge_attacker_ids": sorted(self.edge_attacker_ids),
            "initial_accuracy": initial_accuracy,
            "initial_loss": initial_loss,
            "final_accuracy": self.round_records[-1]["global_accuracy"],
            "best_accuracy": self.best_acc,
            "last_5_round_accuracy": float(np.mean([row["global_accuracy"] for row in tail])),
            "pre_attack_accuracy": pre_attack_accuracy,
            "attack_min_accuracy": attack_min_accuracy,
            "max_attack_accuracy_drop": (
                max(0.0, pre_attack_accuracy - attack_min_accuracy) if attack_min_accuracy is not None else None
            ),
            "recovery_round": recovery_round,
            "recovery_delay": None if recovery_round is None else recovery_round - start_display,
            "target_accuracy": target,
            "time_to_target_round": time_to_target,
            "cumulative_used_samples": cumulative_used_samples,
            "cumulative_energy": cumulative_energy,
            "cumulative_client_payment": cumulative_payment,
            "cumulative_malicious_payment": cumulative_malicious_payment,
            "malicious_payment_share": cumulative_malicious_payment / max(cumulative_payment, 1e-12),
            "post_attack_malicious_payment_share": (
                post_attack_malicious_payment / max(post_attack_payment, 1e-12)
            ),
            "detection_round": detection_round,
            "detection_delay": None if detection_round is None else detection_round - start_display,
            "final_mean_remaining_energy": self.round_records[-1]["mean_remaining_energy"],
            "final_reputation_gap": (
                self.round_records[-1]["mean_honest_reputation"]
                - self.round_records[-1]["mean_malicious_reputation"]
                if malicious_reps and honest_reps else None
            ),
            "final_malicious_detection_rate": self.round_records[-1]["malicious_detection_rate"],
            "final_honest_false_positive_rate": self.round_records[-1]["low_reputation_honest_rate"],
        }
        self._save_results(summary)
        return summary
