import math
import random


import numpy as np
import torch
from torch.functional import F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from tqdm import tqdm
import copy
from src.models.model import choose_model
from src.optimizers.gd import GD
from src.group_module.client_evaluator import *
from src.fed_client.fedgroup_client import FedGroupClient
from src.group_module.group_maker import Group_Maker
from src.fed_client.base_client import Client
from src.cost import ClientAttr, Cost
from src.utils.metrics import Metrics
from src.allocation_alo.bandwidth_allocation import Bandwidth_Allocation
from src.fed_MEC.MEC_server import MEC_server
from src.fed_client.Client_server import Client_server
import torch.optim as optim
from src.fuzzy import fuzzy

np.random.seed(2025)
class base_server:
    def __init__(self, options, dataset, clients_label):
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.model = choose_model(options)
        self.optimizer = GD(self.model.parameters(), lr=options['lr'])
        # self.optimizer = optim.SGD(self.model.parameters(), lr=options['lr'])
        self.latest_global_model=self.get_model_parameters()
        self.all_clients=self.get_clients(self.dataset, self.clients_label)
        self.selected_clients=[]
        self.metrics = Metrics(options, self.all_clients)
        self.cost = Cost()


    def get_clients(self, dataset, clients_label):
        np.random.seed(2025)
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        clients_server = []
        number=self.options['num_of_clients']
        # Shuffle the clients_label indices
        shuffled_labels = copy.deepcopy(clients_label)
        random.shuffle(shuffled_labels)

        # Generate distances for clients within 200m of the edge server
        location_x=np.random.uniform(0,self.options['space'],number)
        location_y=np.random.uniform(0,self.options['space'],number)
        for i in range(number):
            clients_server.append(Client_server(self.options,i,self.model,self.optimizer,
                TensorDataset(
                    torch.tensor(train_data[shuffled_labels[i]]),
                    torch.tensor(train_label[shuffled_labels[i]])),
                location_x[i],
                location_y[i]
            ))
        return clients_server
    #获得云服务器的模型参数
    def get_model_parameters(self):
        return copy.deepcopy(self.model.state_dict())
    #设置云服务器的模型参数
    def set_model_parameters(self,model_parameters):
        self.model.load_state_dict(model_parameters)
    #将云服务器的模型参数传递给边缘服务器
    def set_parameters_to_client(self):
        for _,client in enumerate(self.selected_clients):
            client.set_parameters(self.latest_global_model)
    def train(self):
        #打印一些调试信息
        # for _,server in enumerate(self.MEC_servers):
        #     for _,client in enumerate(server.Clients_server):
        #         # print(client.id)
        #         # print(":\n",client.cpu,client.power)
        #         print("**",client.local_dataset)
        #     print("\n\n")
        #全局迭代次数
        for round_i in tqdm(range(self.options['round_num'])):
            # Calculate each client's total capability for federated learning
            # based on computational power, communication ability, and data quality
            for client in self.all_clients:
                client.output = fuzzy(client.f, client.com_r(), len(client.local_dataset))
            # Sort clients by output value in descending order
            self.all_clients.sort(key=lambda x: x.output, reverse=True)
            self.selected_clients = self.all_clients[:int(len(self.all_clients) * 0.5)]
            #测试每轮全局的模型准确率
            self.test_latest_model_on_testdata(round_i)
            #进行stackelberg博弈，得到选中客户端的参与意愿和服务器对于每个客户端的单位定价价格
            self.stackelberg()

            parameters=[]
            for _,client in enumerate(self.selected_clients):
                client.set_parameters(self.latest_global_model)
                parameters.append(client.local_train())
            self.aggregate_parameters(parameters)
        #在训练结束后测试最终的全局模型在测试集上的性能
        accuracy,loss= self.test_latest_model_on_testdata(self.options['round_num'])
        #保存指标
        # self.metrics.write()

    def test_latest_model_on_testdata(self,round_i):
        pre_model_paramerters=copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.latest_global_model)
        self.model.eval()
        test_data = torch.tensor(self.dataset.testData)
        test_label = torch.tensor(self.dataset.testLabel)
        if self.options['gpu'] >= 0:
            # test_data, test_label = test_data.cuda(), test_label.cuda()
            test_data, test_label = test_data.cpu(), test_label.cpu()
        with torch.no_grad():
            pred = self.model(test_data)
            loss = F.cross_entropy(pred, test_label)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(test_label).sum().item()
            accuracy = correct / len(test_label)

        self.model.load_state_dict(pre_model_paramerters)
        eval_stats = {
            'loss': loss,
            'acc': accuracy
        }
        print("Global:Loss={:.4f}\nAccuracy={:.4f}".format(loss, accuracy))
        # self.metrics.update_test_stats(round_i, eval_stats)
        return accuracy,loss

    def aggregate_parameters(self,parameters):
        #全局聚合根据数据量聚合
        total_samples = sum([int(len(client.local_dataset)) for client in self.selected_clients])
        averaged_params = {}

        # Get keys from the first client's parameters (assuming all have same structure)
        if len(parameters) > 0:
            for key in parameters[0].keys():
                averaged_params[key] = torch.zeros_like(parameters[0][key])

                # Weighted average of parameters from all clients
                for i, client in enumerate(self.selected_clients):
                    weight = int(len(client.local_dataset)) / total_samples
                    averaged_params[key] += weight * parameters[i][key]
        self.latest_global_model=averaged_params
        self.set_model_parameters(averaged_params)

