"""Client implementation for the Chapter 4 hierarchical-FL experiments."""
from __future__ import annotations

import copy
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class Client_server:
    def __init__(self, options, id, model, optimizer, local_dataset, location_x, location_y):
        self.options = options
        self.id = int(id)
        self.local_dataset = local_dataset
        self.model = model
        self.optimizer = optimizer
        self.MEC = None
        self.x = float(location_x)
        self.y = float(location_y)

        rng = np.random.default_rng(int(options.get("seed", 0)) + 1009 * (self.id + 1))
        self.f = float(rng.uniform(options["f_min"], options["f_max"]))
        self.p = float(rng.uniform(options["p_min"], options["p_max"]))
        self.d = 0.0
        self.h = 0.0
        self.p_r = 0.0
        self.B = 0.0

        self.data_size = len(local_dataset)
        self.max_data_visits = self.data_size * int(options.get("local_epoch", 1))
        self.data_label = len(self.get_label_counts())
        self.data_distribution = self.get_distribution()
        self.distribution = self.ditribution()
        self.class_distribution, self.class_is_own = self.get_class()
        self.output = None
        self.parameters = None

        # Chapter 4 state.
        initial_rep = float(options.get("initial_reputation", 0.5))
        self.history_reputation = initial_rep
        self.instant_weight = initial_rep
        self.previous_update_vector = None
        # Reputation uses the change of the local average gradient, not the
        # difference between two uploaded models.
        self.previous_gradient_vector = None
        self.current_gradient_vector = None
        self.previous_loss = None
        self.last_beta = float(options.get("beta0", 0.9))
        self.last_sigma = 0.0

        self.initial_energy = float(options.get("initial_energy", 25000.0))
        self.current_energy = self.initial_energy
        self.cost_per_unit = 1.0
        self.fixed_cost_per_unit = None
        self.contribution_ratio = 0.0
        self.planned_samples = 0
        self.used_samples = 0
        self.max_samples_by_energy = 0
        self.price_from_edge = 0.0
        self.global_base_price = 0.0
        self.satisfaction_param = 1.0
        self.last_train_acc = 0.0
        self.last_train_loss = None
        self.last_train_stats: Dict[str, float] = {}
        self.last_compute_energy = 0.0
        self.last_communication_energy = 0.0
        self.last_total_energy = 0.0
        self.last_payment = 0.0
        self.last_utility = 0.0
        self.is_byzantine = False
        self.round_gradient_sum = None
        self.round_gradient_weight = 0
        self.round_loss_sum = 0.0
        self.round_loss_weight = 0

    @property
    def device(self):
        gpu = int(self.options.get("gpu", -1))
        if gpu >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cpu")

    def get_model_parameters(self):
        return self.get_parameters()

    def set_model_parameters(self, model_parameters_dict):
        self.set_parameters(model_parameters_dict)

    def get_parameters(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_parameters(self, parameters):
        current = self.model.state_dict()
        state = {k: parameters[k].detach().to(dtype=v.dtype, device=v.device) for k, v in current.items()}
        self.model.load_state_dict(state, strict=True)

    def get_label_counts(self):
        counts = {}
        for _, label in self.local_dataset:
            value = int(label.item()) if torch.is_tensor(label) else int(label)
            counts[value] = counts.get(value, 0) + 1
        return counts

    def get_size(self):
        return len(self.local_dataset)

    def get_label(self):
        return len(self.get_label_counts())

    def get_distribution(self):
        counts = self.get_label_counts()
        total = max(sum(counts.values()), 1)
        return -sum((n / total) * math.log(n / total, 2) for n in counts.values() if n)

    def ditribution(self):  # Retain the legacy misspelled public method.
        class_num = 47 if self.options.get("dataset_name") == "emnist" else 10
        result = np.zeros(class_num, dtype=np.float64)
        counts = self.get_label_counts()
        total = max(sum(counts.values()), 1)
        for label, count in counts.items():
            if 0 <= label < class_num:
                result[label] = count / total
        return result

    def get_class(self):
        counts = self.get_label_counts()
        class_num = 47 if self.options.get("dataset_name") == "emnist" else 10
        distribution = [counts.get(i, 0) for i in range(class_num)]
        return distribution, [int(n > 0) for n in distribution]

    # Communication and energy model -------------------------------------------------
    def get_d(self):
        if self.MEC is None:
            raise RuntimeError(f"client {self.id} has not been assigned to an MEC server")
        self.d = max(math.hypot(self.x - self.MEC.x, self.y - self.MEC.y), 1.0)
        return self.d

    def com_h(self):
        # Distance-normalized path gain.  The lower bound avoids an infinite gain.
        self.h = max(self.get_d(), 1.0) ** (-float(self.options.get("loss_exponent", 3.0)))
        return self.h

    def com_p_r(self):
        self.p_r = self.p * self.com_h()
        return self.p_r

    def com_r(self):
        active = getattr(self.MEC, "active_clients", None) or self.MEC.selected_clients
        users = max(len(active), 1)
        self.B = max(float(self.MEC.B) / users, 1.0)
        noise = max(float(self.options.get("noise_power", 4e-21)) * self.B, 1e-30)
        snr = max(self.com_p_r() / noise, 0.0)
        return max(self.B * math.log2(1.0 + snr), 1e-12)

    def com_energy_trans(self):
        return self.p * float(self.options["model_size"]) / self.com_r()

    def com_energy_com(self, num_samples: Optional[int] = None):
        # q is the number of sample visits; repeated visits are counted.
        q = self.max_data_visits if num_samples is None else max(int(num_samples), 0)
        return (
            float(self.options["coefficient"])
            * self.f**2
            * q
            * float(self.options["CPU_cycles"])
        )

    def com_delay_trans(self):
        return float(self.options["model_size"]) / self.com_r()

    def com_delay_com(self, num_samples: Optional[int] = None):
        q = self.max_data_visits if num_samples is None else max(int(num_samples), 0)
        return q * float(self.options["CPU_cycles"]) / self.f

    def com_energy_total(self, num_samples: Optional[int] = None):
        q = self.max_data_visits if num_samples is None else max(int(num_samples), 0)
        if q == 0:
            return 0.0
        theta = float(self.options.get("theta", 1.0))
        return theta * (self.com_energy_com(q) + self.com_energy_trans())

    def max_feasible_samples(self, remaining_uploads=None):
        """Hard energy constraint Q_i^max(t), including remaining uploads."""
        theta = float(self.options.get("theta", 1.0))
        uploads = int(self.options.get("local_num", 1)) if remaining_uploads is None else int(remaining_uploads)
        communication = max(uploads, 0) * theta * self.com_energy_trans()
        per_sample = theta * self.com_energy_com(1)
        if self.current_energy <= communication or per_sample <= 0:
            self.max_samples_by_energy = 0
        else:
            energy_limit = int(math.floor((self.current_energy - communication) / per_sample))
            deadline = float(self.options.get("round_deadline", 0.0))
            time_limit = (
                int(math.floor(
                    max(deadline - max(uploads, 0) * self.com_delay_trans(), 0.0)
                    * self.f / float(self.options["CPU_cycles"])
                ))
                if deadline > 0 else self.max_data_visits
            )
            self.max_samples_by_energy = min(self.max_data_visits, energy_limit, time_limit)
        return self.max_samples_by_energy

    def update_dynamic_cost(self, fixed=False):
        self.max_feasible_samples()
        communication = (
            int(self.options.get("local_num", 1))
            * float(self.options.get("theta", 1.0))
            * self.com_energy_trans()
        )
        available = max(self.current_energy - communication, 1e-12)
        proposed = float(self.options.get("cost_scale", 1.0)) / available
        if fixed:
            if self.fixed_cost_per_unit is None:
                self.fixed_cost_per_unit = proposed
            self.cost_per_unit = self.fixed_cost_per_unit
        else:
            self.cost_per_unit = proposed
        return self.cost_per_unit

    def plan_contribution(self, requested_samples, remaining_uploads=None):
        qmax = self.max_feasible_samples(remaining_uploads=remaining_uploads)
        q = min(max(int(math.floor(float(requested_samples))), 0), self.max_data_visits, qmax)
        if 0 < q < int(self.options.get("min_train_samples", 1)):
            q = 0
        self.planned_samples = q
        self.contribution_ratio = q / self.max_data_visits if self.max_data_visits else 0.0
        return q

    def settle_energy(self):
        if self.used_samples <= 0:
            self.last_compute_energy = self.last_communication_energy = self.last_total_energy = 0.0
            return 0.0
        theta = float(self.options.get("theta", 1.0))
        self.last_compute_energy = theta * self.com_energy_com(self.used_samples)
        self.last_communication_energy = theta * self.com_energy_trans()
        required = self.last_compute_energy + self.last_communication_energy
        # A one-unit numerical tolerance keeps the hard constraint from drifting.
        if required > self.current_energy + max(1e-9, abs(self.current_energy) * 1e-10):
            raise RuntimeError(f"client {self.id} violated its hard energy constraint")
        self.last_total_energy = min(required, self.current_energy)
        self.current_energy = max(0.0, self.current_energy - self.last_total_energy)
        return self.last_total_energy

    # Local optimization -------------------------------------------------------------
    def local_train(self, round_index=0, local_step=0):
        start = time.perf_counter()
        parameters, stats = self.local_update(round_index=round_index, local_step=local_step)
        stats["time"] = time.perf_counter() - start
        self.last_train_stats = stats
        if stats["used_samples"] > 0:
            self.last_train_loss = stats["loss"]
            self.last_train_acc = stats["acc"]
            # A global round may contain multiple edge-local steps.  Aggregate
            # observations now; the server commits them after the round.
            if self.current_gradient_vector is not None:
                contribution = self.current_gradient_vector * float(stats["used_samples"])
                self.round_gradient_sum = contribution if self.round_gradient_sum is None else self.round_gradient_sum + contribution
                self.round_gradient_weight += int(stats["used_samples"])
            self.round_loss_sum += float(stats["loss"]) * float(stats["used_samples"])
            self.round_loss_weight += int(stats["used_samples"])
        return parameters

    def local_update(self, round_index=0, local_step=0):
        total = len(self.local_dataset)
        total_visits = self.max_data_visits
        actual = min(int(self.planned_samples), total_visits)
        self.used_samples = actual
        if actual <= 0:
            self.current_gradient_vector = None
            return self.get_parameters(), {
                "id": self.id,
                "loss": self.last_train_loss if self.last_train_loss is not None else 0.0,
                "acc": self.last_train_acc,
                "used_samples": 0,
                "total_samples": total_visits,
                "contribution_ratio": 0.0,
            }

        generator = torch.Generator()
        generator.manual_seed(
            int(self.options.get("seed", 0)) + 100003 * int(round_index) + 1009 * int(local_step) + self.id
        )
        # Build exactly q sample visits.  Full local epochs are shuffled; a final
        # partial epoch implements the integer KKT response without over-counting.
        indices = []
        remaining = actual
        while remaining > 0:
            take = min(remaining, total)
            indices.extend(torch.randperm(total, generator=generator)[:take].tolist())
            remaining -= take
        subset = Subset(self.local_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=min(int(self.options["batch_size"]), actual),
            shuffle=True,
            generator=generator,
        )

        device = self.device
        self.model.to(device)
        self.model.train()
        total_loss = total_correct = total_seen = 0
        gradient_sum = None
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            # Store the sample-weighted mean gradient for the PDF's Δg term.
            batch_gradient = torch.cat([
                parameter.grad.detach().cpu().float().reshape(-1)
                for _, parameter in sorted(self.model.named_parameters())
                if parameter.grad is not None
            ])
            gradient_sum = batch_gradient * float(labels.size(0)) if gradient_sum is None else gradient_sum + batch_gradient * float(labels.size(0))
            self.optimizer.step()
            batch = labels.size(0)
            total_loss += float(loss.item()) * batch
            total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            total_seen += batch

        parameters = self.get_parameters()
        self.model.cpu()
        self.current_gradient_vector = gradient_sum / max(total_seen, 1)
        stats = {
            "id": self.id,
            "loss": total_loss / max(total_seen, 1),
            "acc": total_correct / max(total_seen, 1),
            "used_samples": actual,
            "total_samples": total_visits,
            "contribution_ratio": actual / total_visits if total_visits else 0.0,
        }
        if self.options.get("verbose_clients", False):
            print(
                f"client={self.id} used={actual}/{total_visits} "
                f"loss={stats['loss']:.4f} acc={stats['acc']:.4f}"
            )
        return parameters, stats

    def test(self, test_data, test_label):
        device = self.device
        self.model.to(device).eval()
        with torch.no_grad():
            inputs = torch.as_tensor(test_data).to(device)
            labels = torch.as_tensor(test_label).long().to(device)
            logits = self.model(inputs)
            loss = float(F.cross_entropy(logits, labels).item())
            accuracy = float(logits.argmax(1).eq(labels).float().mean().item())
        self.model.cpu()
        return {"loss": loss, "acc": accuracy}
