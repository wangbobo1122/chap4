import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import random

# 设置随机种子保证可重复性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
NUM_CLIENTS = 10  # 总客户端数量
NUM_EDGES = 2  # 边缘服务器数量
NUM_EPOCHS = 5  # 总训练轮次
CLIENT_EPOCHS = 1  # 客户端本地训练轮次
BATCH_SIZE = 64  # 批量大小
LR = 0.01  # 学习率

# 模型分割点配置
SPLIT_LAYER = 3  # 分割层位置（客户端保留前3层）


# 数据加载与分配（使用MNIST数据集）
def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载完整数据集
    full_dataset = datasets.MNIST('./data_temp', train=True, download=True, transform=transform)

    # 非IID数据划分：每个客户端分配2类数字
    client_data = {i: [] for i in range(NUM_CLIENTS)}
    for digit in range(10):
        indices = [i for i, (_, label) in enumerate(full_dataset) if label == digit]
        for idx in indices:
            client_id = digit % NUM_CLIENTS  # 非IID分配
            client_data[client_id].append(idx)

    # 创建客户端数据集
    client_loaders = []
    for i in range(NUM_CLIENTS):
        subset = Subset(full_dataset, client_data[i])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)

    return client_loaders


# 定义分割模型
class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.client_net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.server_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x, mode='full'):
        if mode == 'client':
            return self.client_net(x)
        elif mode == 'server':
            return self.server_net(x)
        else:
            x = self.client_net(x)
            return self.server_net(x)


# 客户端类
class Client:
    def __init__(self, data_loader, client_id, edge_id):
        self.data_loader = data_loader
        self.client_model = SplitModel().client_net.to(device)
        self.optimizer = optim.SGD(self.client_model.parameters(), lr=LR)
        self.edge_id = edge_id
        self.client_id = client_id

    def local_train(self, server_model):
        self.client_model.train()
        server_model = server_model.to(device)

        for _ in range(CLIENT_EPOCHS):
            for data, labels in self.data_loader:
                data, labels = data.to(device), labels.to(device)

                # 前向传播到分割层
                client_output = self.client_model(data)

                # 分离计算图以便服务器端反向传播
                client_output = client_output.detach().requires_grad_(True)

                # 服务器端继续计算
                server_output = server_model(client_output)
                loss = nn.CrossEntropyLoss()(server_output, labels)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 将服务器梯度传回客户端模型
                client_output.backward(client_output.grad)

                # 更新客户端模型参数
                self.optimizer.step()

        return self.client_model.state_dict()


# 边缘服务器类
class EdgeServer:
    def __init__(self, clients):
        self.clients = clients  # 隶属的客户端列表
        self.server_model = SplitModel().server_net.to(device)
        self.optimizer = optim.SGD(self.server_model.parameters(), lr=LR)

    def aggregate_clients(self, client_params_list):
        # 简单平均聚合客户端模型参数
        avg_params = {}
        for key in client_params_list[0].keys():
            avg_params[key] = torch.stack([params[key].float() for params in client_params_list]).mean(dim=0)
        return avg_params

    def train_round(self, global_client_params):
        # 下发全局客户端模型参数
        client_params_list = []
        for client in self.clients:
            # 客户端加载全局参数
            client.client_model.load_state_dict(global_client_params)
            # 本地训练
            local_params = client.local_train(self.server_model)
            client_params_list.append(local_params)

        # 聚合客户端模型参数
        aggregated_params = self.aggregate_clients(client_params_list)
        return aggregated_params


# 中央服务器类
class CentralServer:
    def __init__(self, edge_servers):
        self.edge_servers = edge_servers
        self.global_client_model = SplitModel().client_net.state_dict()

    def aggregate_edges(self, edge_params_list):
        # 简单平均聚合边缘服务器参数
        avg_params = {}
        for key in edge_params_list[0].keys():
            avg_params[key] = torch.stack([params[key].float() for params in edge_params_list]).mean(dim=0)
        return avg_params

    def global_update(self):
        edge_params_list = []
        for edge in self.edge_servers:
            # 边缘服务器进行训练轮次
            edge_params = edge.train_round(self.global_client_model)
            edge_params_list.append(edge_params)

        # 聚合边缘服务器参数
        self.global_client_model = self.aggregate_edges(edge_params_list)
        return self.global_client_model


# 训练流程
def main():
    # 准备数据
    client_loaders = prepare_data()

    # 创建客户端实例（分配给边缘服务器）
    clients = []
    edge_servers = []
    clients_per_edge = NUM_CLIENTS // NUM_EDGES
    for edge_id in range(NUM_EDGES):
        edge_clients = [
            Client(client_loaders[i], client_id=i, edge_id=edge_id)
            for i in range(edge_id * clients_per_edge, (edge_id + 1) * clients_per_edge)
        ]
        edge = EdgeServer(edge_clients)
        edge_servers.append(edge)
        clients.extend(edge_clients)

    # 初始化中央服务器
    central = CentralServer(edge_servers)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        # 全局模型更新
        global_params = central.global_update()

        # 测试全局模型性能（可选）
        # 注意：实际联邦学习中不应该访问测试数据，这里仅为演示
        test_model = SplitModel().to(device)
        test_model.client_net.load_state_dict(global_params)
        test_model.server_net = edge_servers[0].server_model

        # 测试代码（需要单独准备测试数据集）
        # ...

        print(f"Epoch {epoch + 1} completed")


if __name__ == "__main__":
    main()