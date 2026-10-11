#用于数据的序列化和反序列化
import pickle
import json
import numpy as np
import os
import time
#用于图像预处理
import torchvision.transforms as transforms
#用于记录训练和评估过程中的指标，方便可视化
from tensorboardX import SummaryWriter
#用于定义数据集
from torch.utils.data import Dataset
from PIL import Image
import random
import torch


#创建目录，返回目录路径
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

#用于记录联邦学习过程中的各种指标，包括通信、计算、延迟、能量消耗
class Metrics(object):
    def __init__(self, options, clients, name=''):
        self.options = options

        num_rounds = options['round_num'] + 1
        #用于记录每一轮次写入的字节数、计算量、读取的字节数（对于进行计算的客户端）
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        # global_test_data
        self.loss_on_g_test_data = [0] * num_rounds
        self.acc_on_g_test_data = [0] * num_rounds 
        # local and upload e and d
        self.local_latency = [0] * num_rounds 
        self.local_energy = [0] * num_rounds 

        self.upload_latency = [0] * num_rounds 
        self.upload_energy = [0] * num_rounds 
        self.D = []

        # cost time and delay
        self.accumulation_delay = [0] * num_rounds
        self.accumulation_energy = [0] * num_rounds

        #记录每一轮边缘服务器的acc和loss
        self.local_acc=[0]*3001
        self.local_loss=[0]*3001
        #记录每一轮的效用
        self.utility=[0]*num_rounds
        self.jsd=[0]*3001
        self.total=[0]*3001
        self.average_utility_of_clients=[0]*3001
        #记录每个联盟的最终效用
        self.utility_array=[0]*self.options['num_of_MEC']
        #记录每个边缘服务器的最终效用
        self.total_utility_of_MEC=[0]*self.options['num_of_MEC']

        #记录最终选中的每个边缘服务器的迭代次数
        self.total_k=[0]*self.options['num_of_MEC']

        #记录最终的R值
        self.R=0
        if self.options['method_division'] == 1:
            self.result_path = mkdir(os.path.join('./result/dirichlet', str(self.options['dataset_name']).lower() + str(self.options['dirichlet'])))
        elif self.options['method_division'] == 2:
            self.result_path = mkdir(os.path.join('./result/pathology', str(self.options['dataset_name']).lower()))
        elif self.options['method_division'] == 3:
            self.result_path = mkdir(os.path.join('./result/split_iid', str(self.options['dataset_name']).lower()))
        elif self.options['method_division'] == 4:
            self.result_path = mkdir(os.path.join('./result/split', str(self.options['dataset_name']).lower()))
        elif self.options['method_division'] == 5:
            self.result_path = mkdir(os.path.join('./result/split_NIID', str(self.options['dataset_name']).lower()))
        if self.options['batch_size'] == 0:
            suffix = '{}_sd{}_lr{}_ne{}_method{}_lamuda{}'.format(name,
                                            options['seed'],
                                            options['lr'],
                                            options['round_num'],
                                            options['method'],
                                            options['lamuda']
                                               )
        else:
            suffix = '{}_sd{}_lr{}_ne{}_bs{}_w{}_method{}_lamuda{}'.format(name,
                                            options['seed'],
                                            options['lr'],
                                            options['round_num'],
                                            options['batch_size'],
                                            options['weight'],
                                            options['method'],
                                            options['lamuda']
                                                )
        # self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algorithm'],
        #                                      options['model_name'], suffix)
        self.exp_name = '{}_{}_{}_{}'.format(options['server'],
                                             options['model_name'], options['opti'], suffix)

        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        test_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(test_event_folder)

    #更换客户端的通信统计信息
    def update_communication_stats(self, round_i, stats):
        #客户端id、写入的字节数、计算量、读取的字节数
        id, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']
        self.bytes_written[id][round_i] += bytes_w
        self.client_computations[id][round_i] += comp
        self.bytes_read[id][round_i] += bytes_r
    #批量更新多个客户端的通信统计信息，通过调用update_communication_stats
    def extend_communication_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_communication_stats(round_i, stats)
    #跟新测试数据的统计信息，传入参数为当前的轮次以及测试数据的损失和准确率的字典
    def update_test_stats(self, round_i, eval_stats):
        self.loss_on_g_test_data[round_i] = eval_stats['loss']
        self.acc_on_g_test_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)
    #更新延迟和能量消耗的统计信息
    #当前的轮次、本地计算的延迟、上传数据的延迟、累计的延迟、本地计算的能量消耗、上传数据的能量消耗、累计的能量消耗
    def update_cost(self, round_i, local_latency, upload_latency, accumulated_latency,\
                                    local_energy, upload_energy, accumulated_energy):
        self.accumulation_delay[round_i] = accumulated_latency
        self.local_latency[round_i] = local_latency
        self.upload_latency[round_i] = upload_latency
        self.accumulation_energy[round_i] = accumulated_energy
        self.local_energy[round_i] = local_energy
        self.upload_energy[round_i] = upload_energy
    #更新类别的数量信息
    def update_class_vloume_round(self, D):
        self.D.append(D)
    #将所有记录的指标保存到JSON文件中
    def write(self):
        metrics = dict()
        # 保存数据集的名称
        metrics['dataset'] = self.options['dataset_name']
        metrics['local_acc']=self.local_acc
        metrics['local_loss']=self.local_loss
        # 将每个标量 Tensor 提取为单个值
        self.loss_on_g_test_data = [tensor.item() for tensor in self.loss_on_g_test_data]
        metrics['loss_on_g_test_data'] = self.loss_on_g_test_data
        # 保存测试数据的准确率
        metrics['acc_on_g_test_data'] = self.acc_on_g_test_data
        #确定保存路径，保存的根目录、实验名称、保存的文件名
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')
        #将metrics字典写入JSON文件，indent=8表示JSON文件中的内容会以8个空格缩进，使文件更易读
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf, indent=8)