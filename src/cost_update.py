
import math


class Cost_update(object):
    def __init__(self):
        #用于记录基站、无人机和终端设备的总效用
        self.total_utility_of_BS=0
        self.total_utility_of_UAV=0
        self.total_utility_of_ED=0
        #定义计算各种
        self.c_comp=0.5
        self.c_trans=100
        self.c_comp_uav=2.5
        self.c_trans_uav=200
        self.g=0.01

    #计算传输速率R
    def getTransmit(self):
        R=0
        return R

    def get_BS(self,N,stats,D,P):
        local_utility_of_BS=0
        for i in range(N):
            local_utility_of_BS+=(stats[i]['acc']*(math.log(1+D[i]))-P[i]*D[i])
        return local_utility_of_BS

    def get_UAV(self,N,P,Q,D,options):
        local_utility_of_UAV=0
        for i in range(N):
            local_utility_of_UAV+=(P[i]-Q[i])*D[i]-c_comp_uav*N*options['model_size']-c_trans_uav*options['model_size']/10000
        return local_utility_of_UAV

    def get_ED(self,N,Q,D,options,selected_client,round_i,bandwidth):
        local_utility_of_ED=0
        for i in range(N):
            local_utility_of_ED+=Q[i]*D[i]-c_comp*D[i]-c_trans*options['model_size']/selected_client[i].get_R(round_i,bandwidth)-self.g*D[i]
        return local_utility_of_ED


class ClientAttr(object):
    def __init__(self, cpu_frequency, B, transmit_power):
        self.cpu_frequency = cpu_frequency
        self.bandwidth = B
        self.transmit_power = transmit_power

    def get_client_attr(self, id):
        return {
            "cpu_frequency": self.cpu_frequency[id],
            "B": self.bandwidth,
            "transmit_power": self.transmit_power[id],
        }