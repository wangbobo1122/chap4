from src.models.cifar10_alexnet import CIFAR10_AlexNet
from src.models.mnist_dnn import Mnist_DNN
from src.models.mnist_cnn import Mnist_CNN
from src.models.fmnist_cnn import FMnist_CNN
from src.models.cifar10_vgg16 import CIFAR10_VGG16
from src.models.cifar10_cnn import CIFAR10_CNN
from src.models.emnist_cnn import EMNISTCNN
from src.models.emnist_dnn import EMNISTDNN
from src.models.cinic10_cnn import CINIC10_CNN
import torch
import torch.nn as nn
import numpy as np


def choose_model(options):
    model_name = str(options['model_name']).lower()
    # Randomness is initialized once by main.setup_seed; resetting it here
    # would silently make --seed ineffective for model initialization.
    if model_name == 'mnist_dnn':

        # for name, param in Mnist_CNN().named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        return Mnist_DNN()
    if model_name == 'mnist_cnn':
        return Mnist_CNN()
    elif model_name == 'fmnist_cnn':
        return FMnist_CNN()
    elif model_name == 'emnist_cnn':
        return EMNISTCNN()
    elif model_name == 'emnist_dnn':
        return EMNISTDNN()
    elif model_name == 'cifar10_alexnet':
        return CIFAR10_AlexNet()
    elif model_name == 'cifar10_vgg16':
        return CIFAR10_VGG16()
    elif model_name == 'cifar10_cnn':
        return CIFAR10_CNN()
    elif model_name == 'cinic10_cnn':
        return CINIC10_CNN()



