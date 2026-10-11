import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from main import model_payload_bits
from src.fed_client.Client_server import Client_server
from src.fed_server.hierarchical_server import Hierarchical_server
from src.models.mnist_cnn import Mnist_CNN
from src.optimizers.gd import GD
from src.utils.imbanlance import (
    dirichlet_split_noniid,
    split,
    split_NIID,
    split_iid,
    split_imbalance_two,
)


class Chapter4Tests(unittest.TestCase):
    def test_model_payload_is_state_dict_bits(self):
        model = Mnist_CNN()
        expected = sum(value.numel() * value.element_size() * 8 for value in model.state_dict().values())
        self.assertEqual(model_payload_bits(model), expected)

    def test_dirichlet_partition_is_complete_unique_and_reproducible(self):
        labels = np.arange(120) % 10
        first, _ = dirichlet_split_noniid(labels, 0.2, 8, seed=42)
        second, _ = dirichlet_split_noniid(labels, 0.2, 8, seed=42)
        flat = np.concatenate(first)
        self.assertEqual(len(flat), len(labels))
        self.assertEqual(len(np.unique(flat)), len(labels))
        self.assertEqual(sorted(flat.tolist()), list(range(len(labels))))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))

    def test_all_cli_splitters_preserve_the_dataset(self):
        labels = np.arange(200) % 10
        splitters = [
            lambda: split_imbalance_two(labels, 8, seed=5),
            lambda: split_iid(labels, 8, seed=5),
            lambda: split(labels, 8, seed=5),
            lambda: split_NIID(labels, 0.2, 8, seed=5),
        ]
        for make_partition in splitters:
            partition = make_partition()
            flat = np.concatenate(partition)
            self.assertEqual(len(partition), 8)
            self.assertEqual(len(flat), len(labels))
            self.assertEqual(len(np.unique(flat)), len(labels))

    def test_truth_discovery_supports_one_model(self):
        server = Hierarchical_server.__new__(Hierarchical_server)
        server.td_epsilon = 1e-6
        server.td_max_iterations = 20
        state = {"weight": torch.tensor([1.0, 2.0])}
        weights, truth, iterations = server._truth_discovery([state], state)
        self.assertEqual(weights, [1.0])
        self.assertEqual(iterations, 1)
        self.assertTrue(torch.equal(truth["weight"], state["weight"]))

    def test_adaptive_ema_uses_sum_of_median_normalized_components(self):
        server = Hierarchical_server.__new__(Hierarchical_server)
        server.options = {"sigma_epsilon": 1e-12, "beta0": 0.9, "attack_sensitivity": 1.0}
        server.use_adaptive_ema = True
        actors = [SimpleNamespace(history_reputation=0.5), SimpleNamespace(history_reputation=0.5)]
        components = [[2.0, 4.0], [1.0, 1.0], [0.5, 0.5]]
        server._ema_reputation_update(actors, [0.2, 0.8], components, [[True, True]] * 3)
        expected_sigma = 2.0 / 3.0 + 1.0 + 1.0
        self.assertAlmostEqual(actors[0].last_sigma, expected_sigma)
        expected_beta = 0.9 * math.exp(-expected_sigma)
        self.assertAlmostEqual(actors[0].last_beta, expected_beta)
        self.assertAlmostEqual(
            actors[0].history_reputation,
            expected_beta * 0.5 + (1.0 - expected_beta) * 0.2,
        )

    def test_unconstrained_edge_price_matches_pdf_equation_4_28(self):
        clients = [
            SimpleNamespace(id=0, history_reputation=0.5, cost_per_unit=2.0, max_samples_by_energy=10_000),
            SimpleNamespace(id=1, history_reputation=0.8, cost_per_unit=4.0, max_samples_by_energy=10_000),
        ]
        edge = SimpleNamespace(active_clients=clients, history_reputation=0.8)
        cloud_price = 10.0
        price, quantities, _, _ = Hierarchical_server._edge_response(edge, cloud_price)
        squared_sum = sum(c.history_reputation ** 2 / c.cost_per_unit for c in clients)
        linear_sum = sum(c.history_reputation / c.cost_per_unit for c in clients)
        expected_price = cloud_price * edge.history_reputation * squared_sum / (2.0 * linear_sum)
        self.assertAlmostEqual(price, expected_price)
        for client in clients:
            self.assertAlmostEqual(
                quantities[client.id], price * client.history_reputation / client.cost_per_unit
            )

    def test_energy_equations_use_bits_and_respect_qmax(self):
        options = {
            "seed": 1, "f_min": 2e9, "f_max": 2e9, "p_min": 0.2, "p_max": 0.2,
            "local_epoch": 1, "initial_reputation": 0.5, "beta0": 0.9,
            "initial_energy": 100.0, "cost_scale": 2.0, "coefficient": 1e-28,
            "CPU_cycles": 1e7, "model_size": 8e6, "noise_power": 4e-21,
            "loss_exponent": 3.0, "theta": 1.0, "local_num": 1,
            "round_deadline": 0.0, "min_train_samples": 1, "gpu": -1,
            "dataset_name": "mnist", "batch_size": 2,
        }
        model = torch.nn.Linear(1, 2)
        dataset = torch.utils.data.TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))
        client = Client_server(options, 0, model, GD(model.parameters(), lr=0.1), dataset, 0.0, 0.0)
        client.MEC = SimpleNamespace(x=1.0, y=0.0, B=1e6, active_clients=[client], selected_clients=[client])
        communication = client.com_energy_trans()
        per_sample = options["coefficient"] * client.f ** 2 * options["CPU_cycles"]
        expected = max(0, math.floor((client.current_energy - communication) / per_sample))
        self.assertEqual(client.max_feasible_samples(), min(client.max_data_visits, expected))
        self.assertAlmostEqual(client.update_dynamic_cost(), options["cost_scale"] / (client.current_energy - communication))

    def test_fixed_budget_is_proportional_and_respects_energy_caps(self):
        server = Hierarchical_server.__new__(Hierarchical_server)
        server.options = {"round_sample_budget": 11}
        clients = [
            SimpleNamespace(id=0, data_size=20, max_samples_by_energy=10),
            SimpleNamespace(id=1, data_size=30, max_samples_by_energy=10),
            SimpleNamespace(id=2, data_size=50, max_samples_by_energy=10),
        ]
        quantities = server._allocate_fixed_budget(clients)
        self.assertEqual(quantities, {0: 2.0, 1: 3.0, 2: 6.0})
        self.assertEqual(sum(quantities.values()), 11)

        clients[0].max_samples_by_energy = 1
        quantities = server._allocate_fixed_budget(clients)
        self.assertEqual(quantities[0], 1.0)
        self.assertEqual(sum(quantities.values()), 11)


if __name__ == "__main__":
    unittest.main()
