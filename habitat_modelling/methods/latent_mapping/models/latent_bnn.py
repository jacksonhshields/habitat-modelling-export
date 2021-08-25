import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class FCNet1(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(FCNet1, self).__init__()
        # ----------------
        # Store Params
        # ----------------
        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes

        # ---------------
        # Create layers
        # ---------------
        in_units = input_dim
        out_units = layer_neurons[0]
        self.fc1 = nn.Linear(in_units, out_units)
        in_units = out_units
        self.out = nn.Linear(in_units, self.num_classes)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)


def init_latent_bnn1_from_params(params):
    return LatentBNN1(**params)

class LatentBNN1(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(LatentBNN1, self).__init__()
        self.net = FCNet1(input_dim, layer_neurons, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softplus = nn.Softplus()

    # def load_state(self, path):
    #     self.net.load_state_dict(torch.load(path))
    #
    # def save_state(self, path):
    #     torch.save(self.net.state_dict(), path)

    def dump_params(self, path):
        self.net.dump_params(path)

    def model(self, x_data, y_data):
        fc1w_prior = Normal(loc=torch.zeros_like(self.net.fc1.weight), scale=torch.ones_like(self.net.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(self.net.fc1.bias), scale=torch.ones_like(self.net.fc1.bias))
        outw_prior = Normal(loc=torch.zeros_like(self.net.out.weight), scale=torch.ones_like(self.net.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(self.net.out.bias), scale=torch.ones_like(self.net.out.bias))

        priors = {
                "fc1.weight": fc1w_prior,
                "fc1.bias": fc1b_prior,
                "out.weight": outw_prior,
                "out.bias": outb_prior
                }

        lifted_module = pyro.random_module("module", self.net, priors)

        lifted_reg_module = lifted_module()

        lhat = self.log_softmax(lifted_reg_module(x_data))

        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def guide(self, x_data, y_data):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(self.net.fc1.weight)
        fc1w_sigma = torch.randn_like(self.net.fc1.weight)
        fc1w_mu_param = pyro.param('fc1w_mu', fc1w_mu)
        fc1w_sigma_param = self.softplus(pyro.param('fc1w_sigma', fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)

        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(self.net.fc1.bias)
        fc1b_sigma = torch.randn_like(self.net.fc1.bias)
        fc1b_mu_param = pyro.param('fc1b_mu', fc1b_mu)
        fc1b_sigma_param = self.softplus(pyro.param('fc1b_sigma', fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # Output layer weight distribution priors
        outw_mu = torch.randn_like(self.net.out.weight)
        outw_sigma = torch.randn_like(self.net.out.weight)
        outw_mu_param = pyro.param('outw_mu', outw_mu)
        outw_sigma_param = self.softplus(pyro.param('outw_sigma', outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)

        # Output layer sigma distribution priors
        outb_mu = torch.randn_like(self.net.out.bias)
        outb_sigma = torch.randn_like(self.net.out.bias)
        outb_mu_param = pyro.param('outb_mu', outb_mu)
        outb_sigma_param = self.softplus(pyro.param('outb_sigma', outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

        priors = {
                "fc1.weight": fc1w_prior,
                "fc1.bias": fc1b_prior,
                "out.weight": outw_prior,
                "out.bias": outb_prior
                }
        lifted_module = pyro.random_module("module", self.net, priors)

        return lifted_module()

    def predict(self, x, num_samples=100):
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        yhats = [F.log_softmax(model(x).data, 1) for model in sampled_models]
        yhats = [yh.detach().cpu().numpy() for yh in yhats]
        return np.asarray(yhats)

class FCNet2(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(FCNet2, self).__init__()
        # ----------------
        # Store Params
        # ----------------
        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes

        # ---------------
        # Create layers
        # ---------------
        in_units = input_dim
        out_units = layer_neurons[0]
        self.fc1 = nn.Linear(in_units, out_units)
        in_units = out_units
        out_units = layer_neurons[1]
        self.fc2 = nn.Linear(in_units, out_units)
        in_units = out_units
        self.out = nn.Linear(in_units, self.num_classes)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.out(output)
        return output

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)


def init_latent_bnn2_from_params(params):
    return LatentBNN1(**params)

class LatentBNN2(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(LatentBNN2, self).__init__()
        self.net = FCNet2(input_dim, layer_neurons, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softplus = nn.Softplus()

    # def load_state(self, path):
    #     self.net.load_state_dict(torch.load(path))
    #
    # def save_state(self, path):
    #     torch.save(self.net.state_dict(), path)

    def dump_params(self, path):
        self.net.dump_params(path)

    def model(self, x_data, y_data):
        fc1w_prior = Normal(loc=torch.zeros_like(self.net.fc1.weight), scale=torch.ones_like(self.net.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(self.net.fc1.bias), scale=torch.ones_like(self.net.fc1.bias))
        fc2w_prior = Normal(loc=torch.zeros_like(self.net.fc2.weight), scale=torch.ones_like(self.net.fc2.weight))
        fc2b_prior = Normal(loc=torch.zeros_like(self.net.fc2.bias), scale=torch.ones_like(self.net.fc2.bias))
        outw_prior = Normal(loc=torch.zeros_like(self.net.out.weight), scale=torch.ones_like(self.net.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(self.net.out.bias), scale=torch.ones_like(self.net.out.bias))

        priors = {
                "fc1.weight": fc1w_prior,
                "fc1.bias": fc1b_prior,
                "fc2.weight": fc2w_prior,
                "fc2.bias": fc2b_prior,
                "out.weight": outw_prior,
                "out.bias": outb_prior
                }

        lifted_module = pyro.random_module("module", self.net, priors)

        lifted_reg_module = lifted_module()

        lhat = self.log_softmax(lifted_reg_module(x_data))

        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def guide(self, x_data, y_data):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(self.net.fc1.weight)
        fc1w_sigma = torch.randn_like(self.net.fc1.weight)
        fc1w_mu_param = pyro.param('fc1w_mu', fc1w_mu)
        fc1w_sigma_param = self.softplus(pyro.param('fc1w_sigma', fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)

        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(self.net.fc1.bias)
        fc1b_sigma = torch.randn_like(self.net.fc1.bias)
        fc1b_mu_param = pyro.param('fc1b_mu', fc1b_mu)
        fc1b_sigma_param = self.softplus(pyro.param('fc1b_sigma', fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # Second layer weight distribution priors
        fc2w_mu = torch.randn_like(self.net.fc2.weight)
        fc2w_sigma = torch.randn_like(self.net.fc2.weight)
        fc2w_mu_param = pyro.param('fc2w_mu', fc2w_mu)
        fc2w_sigma_param = self.softplus(pyro.param('fc2w_sigma', fc2w_sigma))
        fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)

        # Second layer bias distribution priors
        fc2b_mu = torch.randn_like(self.net.fc2.bias)
        fc2b_sigma = torch.randn_like(self.net.fc2.bias)
        fc2b_mu_param = pyro.param('fc2b_mu', fc2b_mu)
        fc2b_sigma_param = self.softplus(pyro.param('fc2b_sigma', fc2b_sigma))
        fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)

        # Output layer weight distribution priors
        outw_mu = torch.randn_like(self.net.out.weight)
        outw_sigma = torch.randn_like(self.net.out.weight)
        outw_mu_param = pyro.param('outw_mu', outw_mu)
        outw_sigma_param = self.softplus(pyro.param('outw_sigma', outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)

        # Output layer sigma distribution priors
        outb_mu = torch.randn_like(self.net.out.bias)
        outb_sigma = torch.randn_like(self.net.out.bias)
        outb_mu_param = pyro.param('outb_mu', outb_mu)
        outb_sigma_param = self.softplus(pyro.param('outb_sigma', outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

        priors = {
                "fc1.weight": fc1w_prior,
                "fc1.bias": fc1b_prior,
                "fc2.weight": fc2w_prior,
                "fc2.bias": fc2b_prior,
                "out.weight": outw_prior,
                "out.bias": outb_prior
                }
        lifted_module = pyro.random_module("module", self.net, priors)

        return lifted_module()

    def predict(self, x, num_samples=100):
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        yhats = [F.log_softmax(model(x).data, 1) for model in sampled_models]
        yhats = [yh.detach().cpu().numpy() for yh in yhats]
        return np.asarray(yhats)
