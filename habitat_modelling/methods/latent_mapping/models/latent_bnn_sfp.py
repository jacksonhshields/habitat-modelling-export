import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import json


def get_beta(epoch_idx, N):
    return 1.0 / N / 100

def get_beta_opts(batch_idx, m, epoch, num_epochs, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta

def elbo(out, y, kl, beta):
    loss = F.cross_entropy(out, y)
    return loss + beta * kl

def normalization_function(x):
    return (x) / torch.sum(x, dim=0)

def calc_uncertainty_normalized(output):
    outputs = []
    for t in range(1):
        prediction = F.softplus(output.cpu())
        prediction = normalization_function(prediction)
        outputs.append(prediction)

    res = np.mean(prediction.numpy(), axis=0)
    p_hat = torch.cat(outputs, 1)
    p_hat = p_hat.numpy()
    T = 1

    aleatoric = np.diag(res) - p_hat.T.dot(p_hat) / p_hat.shape[0]
    tmp = p_hat - res
    epistemic = tmp.T.dot(tmp) / tmp.shape[0]
    return (np.sum(epistemic, keepdims=True)), (np.sum(aleatoric, keepdims=True))

def calc_uncertainity_softmax(output):
    prediction = F.softmax(output, dim = 1)
    results = torch.max(prediction, 1 )
    p_hat = np.array(results[0].cpu().detach())
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
#     epistemic += epistemic
    #print (epistemic)
    aleatoric = np.mean(p_hat * (1-p_hat), axis = 0)
#     aleatoric += aleatoric
    #print (aleatoric)
    return epistemic, aleatoric

class BBBLinearFactorial(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_features, out_features, bias=False):
        super(BBBLinearFactorial, self).__init__()
        self.q_logvar_init = q_logvar_init
        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.p_logvar_init)
        self.eps_weight.data.zero_()

    def forward(self, input):
        raise NotImplementedError()

    def fcprobforward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight ** 2 + self.mu_weight ** 2) / (
                    2 * self.q_logvar_init ** 2) - 0.5
        bias = None
        out = F.linear(input, weight, bias)
        kl = kl_.sum()
        return out, kl



class LatentBNNsfp(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(LatentBNNsfp, self).__init__()
        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)

        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes

        in_units = input_dim
        out_units = layer_neurons[0]
        self.fc0 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, in_units, out_units)
        self.soft0 = nn.Softplus()
        in_units = out_units
        out_units = layer_neurons[1]
        self.fc1 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, in_units, out_units)
        self.soft1 = nn.Softplus()
        in_units = out_units
        out_units = num_classes
        self.output = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, in_units, out_units)

        self.layers = nn.ModuleList([self.fc0, self.soft0, self.fc1, self.soft1])

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, kl_ = layer.fcprobforward(x)
            else:
                x = layer.forward(x)
            # kl += kl_
        x, kl_ = self.output.fcprobforward(x)
        kl += kl_
        logits = x
        return logits, kl

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)





