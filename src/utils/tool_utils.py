
import torch
import math
import numpy as np
import random
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import numpy as np


def paraGeneration(options):
    np.random.seed(2024)
    # CPU clock speed f_
    cpu_frequency = [np.round(np.random.uniform(0.1, 5, size = options['round_num']), decimals=1) for client in range(options['num_of_clients'])]
    # B = np.round(np.random.randint(1, 20, size = options['num_of_clients']), decimals=1)
    np.random.shuffle(cpu_frequency)
    #print(cpu_frequency)
    B = 20
    transmit_power = [np.round(np.random.uniform(1, 10), decimals=1) for i in range(options['num_of_clients'])]
    # print(len(transmit_power))
    return cpu_frequency, B, transmit_power

def paraGeneration_h(options):
    np.random.seed(2025)
    # CPU clock speed f_
    #[1,6]*10**6
    cpu_frequency = [np.round(np.random.uniform(1000000, 6000000), decimals=1) for i in range(options['num_of_clients'])]
    # B = np.round(np.random.randint(1, 20, size = options['num_of_clients']), decimals=1)
    np.random.shuffle(cpu_frequency)
    #print(cpu_frequency)
    #[0.1,0.2]W
    transmit_power = [np.round(np.random.uniform(0.1, 0.2), decimals=3) for i in range(options['num_of_clients'])]
    # np.random.shuffle(transmit_power)
    # print(len(transmit_power))
    return cpu_frequency, transmit_power


