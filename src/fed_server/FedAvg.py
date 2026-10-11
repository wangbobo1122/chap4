import math
import torch
import copy
import numpy as np
from torch.functional import F
from torch.utils.data import TensorDataset
from src.fed_client.base_client import Client
from src.cost import ClientAttr, Cost
from src.utils.metrics import Metrics
from tqdm import tqdm
from src.allocation_alo.bandwidth_allocation import Bandwidth_Allocation
import sympy as sp
from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
from src.fed_UAV.base_UAV import UAV
import cvxpy as cp
from tqdm import tqdm
from src.group_module.client_evaluator import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src.fed_client.fedgroup_client import FedGroupClient
import random
from src.group_module.group_maker import Group_Maker
# from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import numpy as np
import copy
from tqdm import tqdm
np.random.seed(2024)
class FedAvg(BaseFederated):
    def __init__(self,options,dataset,clients_label,cpu_frequency,B,transmit_power):
        model=choose_model(options)
        self.move_model_to_gpu(model,options)
        self.optimizer=GD(model.parameters(),lr=options['lr'])
        super(FedAvg, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model,self.optimizer, )
        self.options['model_size'] = model.get_model_size()
        clients_representer = None
        self.client_evaluator = None
        # clients_representer = self.get_client_representer(self.clients)

    def train(self):
        #打印每轮选择的客户端数量(比例*总的客户端数量)
        N=int(self.per_round_c_fraction * self.clients_num)
        print('=== Select {} clients per round ===\n'.format(N))
        #全局迭代次数
        for round_i in tqdm(range(self.num_round)):
            #在每一轮全局迭代中计算测试集的准确率
            self.test_latest_model_on_testdata(round_i)
            #随机选择N个客户端
            selected_clients=self.random_select(self.clients,N)
            #定义本地迭代次数
            local_epoch=[20]*len(selected_clients)
            #本地训练
            local_model_paras_set, stats = self.local_train(round_i, selected_clients, local_epoch)
            #全局聚合到MEC服务器
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)

            # self.optimizer.adjust_learning_rate(round_i)
            # 调整学习率
            self.optimizer.soft_decay_learning_rate()
        #在训练结束后，测试最终的全局模型在测试集上的性能
        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        #保存指标
        self.metrics.write()

    def random_select(self,clients,N):
        select_clients=[]
        select_numbers=random.sample(list(range(0,len(clients))),N)
        for i in range(N):
            select_clients.append(self.clients[select_numbers[i]])
        return select_clients
