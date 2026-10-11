import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import math
import os
from getdata import GetDataSet
from src.models.mnist_cnn import Mnist_CNN

class EV():
    def __init__(self):
        self.model=Mnist_CNN()
        self.dataset = GetDataSet('mnist')

    def evaluate_model(self,dataset):
        self.model.load_state_dict(torch.load('model_params_jsd.pth', weights_only=True))
        self.model.eval()
        test_data = torch.tensor(dataset.testData)
        test_label = torch.tensor(dataset.testLabel)

        with torch.no_grad():
            outputs = self.model(test_data)
            _, predicted = torch.max(outputs.data, 1)

        total = test_label.size(0)
        correct = (predicted == test_label).sum().item()
        print(f'Overall accuracy: {100 * correct / total:.2f}%')

        classes = np.unique(test_label.numpy())
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)

        for i in range(len(test_label)):
            label = test_label[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

        print('\nClass-wise accuracy:')
        for i in range(len(classes)):
            if class_total[i] > 0:
                print(f'Class {i}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')
            else:
                print(f'Class {i}: No samples found')

        return correct / total

if __name__ == '__main__':
    evaluator = EV()
    evaluator.evaluate_model(evaluator.dataset)
