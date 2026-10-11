"""Edge-server state and aggregation helpers."""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def weighted_state_dict(parameters, weights):
    if not parameters:
        raise ValueError("cannot aggregate an empty model list")
    values = np.asarray(weights, dtype=np.float64)
    values = np.maximum(values, 0.0)
    if not np.isfinite(values).all() or values.sum() <= 0:
        values = np.ones(len(parameters), dtype=np.float64)
    values /= values.sum()
    result = {}
    largest = int(np.argmax(values))
    for key, template in parameters[0].items():
        if torch.is_floating_point(template) or torch.is_complex(template):
            acc = torch.zeros_like(template)
            for state, weight in zip(parameters, values):
                acc.add_(state[key].to(acc.device, acc.dtype), alpha=float(weight))
            result[key] = acc
        else:
            result[key] = parameters[largest][key].clone()
    return result


class MEC_server:
    def __init__(self, options, model, id, optimizer, dataset, clients_label, location, B):
        self.options = options
        self.model = model
        self.id = int(id)
        self.dataset = dataset
        self.optimizer = optimizer
        self.clients_label = clients_label
        self.selected_clients = []
        self.active_clients = []
        self.averaged_params = self.get_model_parameters()
        self.latest_local_model = self.get_model_parameters()
        self.total_sample = 0
        self.total_label = 0
        self.total_distribution = 0.0
        self.x, self.y = map(float, location)
        self.B = float(B)
        self.output = None
        self.E = None
        self.k = int(options.get("local_num", 1))
        self.coalition = None
        self.R = 0.0
        self.budget = 0.0
        self.reward = 0.0
        self.EMD = None
        self.WEMD = None
        self.class_is_own = None

        initial_rep = float(options.get("initial_reputation", 0.5))
        self.history_reputation = initial_rep
        self.instant_weight = initial_rep
        self.previous_update_vector = None
        # Previous edge truth model used by the cloud-side edge anomaly term.
        self.previous_truth_model = None
        self.current_truth_model = None
        self.last_beta = float(options.get("beta0", 0.9))
        self.last_sigma = 0.0
        self.is_byzantine = False

        self.price_to_clients = 0.0
        self.share_from_cloud = 0.0
        self.edge_utility = 0.0
        self.edge_revenue = 0.0
        self.client_payment = 0.0
        self.round_weighted_contribution = 0.0
        self.round_used_samples = 0

    def get_model_parameters(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_parameters(self, parameters):
        current = self.model.state_dict()
        state = {k: parameters[k].detach().to(dtype=v.dtype, device=v.device) for k, v in current.items()}
        self.model.load_state_dict(state, strict=True)
        self.averaged_params = self.get_model_parameters()

    def aggregate_parameters(self, parameters, sample_counts: Sequence[float] | None = None):
        if not parameters:
            return self.get_model_parameters()
        weights = list(sample_counts) if sample_counts is not None else [1.0] * len(parameters)
        self.averaged_params = weighted_state_dict(parameters, weights)
        self.set_model_parameters(self.averaged_params)
        self.latest_local_model = copy.deepcopy(self.averaged_params)
        return self.averaged_params

    def get_label_counts(self):
        counts = {}
        for client in self.selected_clients:
            for _, label in client.local_dataset:
                value = int(label.item()) if torch.is_tensor(label) else int(label)
                counts[value] = counts.get(value, 0) + 1
        return counts

    def get_distribution(self):
        counts = self.get_label_counts()
        class_num = 47 if self.options.get("dataset_name") == "emnist" else 10
        result = np.zeros(class_num, dtype=np.float64)
        total = max(sum(counts.values()), 1)
        for label, count in counts.items():
            if 0 <= label < class_num:
                result[label] = count / total
        return result

    def get_total_sample(self):
        self.total_sample = sum(len(client.local_dataset) for client in self.selected_clients)
        return self.total_sample

    def get_total_label(self):
        self.total_label = len(self.get_label_counts())
        return self.total_label

    def get_total_distribution(self):
        distributions = [client.data_distribution for client in self.selected_clients]
        self.total_distribution = float(np.mean(distributions)) if distributions else 0.0
        return self.total_distribution

    def com_energy(self):
        return sum(client.com_energy_total(client.used_samples) for client in self.active_clients)

    def test_latest_model_on_testdata(self):
        device = torch.device(
            f"cuda:{int(self.options.get('gpu', -1))}"
            if int(self.options.get("gpu", -1)) >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(device).eval()
        test = TensorDataset(
            torch.as_tensor(self.dataset.testData),
            torch.as_tensor(self.dataset.testLabel).long(),
        )
        loader = DataLoader(test, batch_size=max(128, int(self.options.get("batch_size", 32))), shuffle=False)
        total_loss = total_correct = total_seen = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = self.model(inputs)
                batch = labels.size(0)
                total_loss += float(F.cross_entropy(logits, labels).item()) * batch
                total_correct += int(logits.argmax(1).eq(labels).sum().item())
                total_seen += batch
        self.model.cpu()
        return total_correct / max(total_seen, 1), total_loss / max(total_seen, 1)

