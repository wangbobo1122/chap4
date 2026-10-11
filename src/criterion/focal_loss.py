from itertools import batched
from math import gamma

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyparsing import alphas


class FederatedFocalLoss:
    def __init__(self, class_distribution, gamma=0, eps=1e-6):
        self.alpha = torch.FloatTensor(1.0 / (class_distribution + eps))
        self.alpha = self.alpha / self.alpha.sum()
        self.gamma = gamma

    def compute_loss(self, logits, labels):
        log_probs=F.log_softmax(logits,dim=1)
        # print(log_probs)
        probs=torch.exp(log_probs)
        batch_size,num_classes=logits.shape


        targets_one_hot=F.one_hot(labels,num_classes).float()
        probs_true=(probs*targets_one_hot).sum(dim=1)
        log_probs_true=(log_probs*targets_one_hot).sum(dim=1)

        modulating_factor=(1.0-probs_true)**self.gamma

        ce_loss=-log_probs_true
        # print(num_classes)
        # print(self.alpha)
        focal_loss=self.alpha[labels]*modulating_factor*ce_loss
        return focal_loss.mean()