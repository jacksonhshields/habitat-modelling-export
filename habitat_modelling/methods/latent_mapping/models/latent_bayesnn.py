import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
import json
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine
import pyro.optim as pyroopt
import pyro.distributions as dist
import pyro.contrib.bnn as bnn
from pyro.infer import SVI, TraceMeanField_ELBO
from torch.distributions import constraints


class BayesNN3(nn.Module):
    """
    Example from: https://alsibahi.xyz/snippets/2019/06/15/pyro_mnist_bnn_kl.html
    """
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(BayesNN3, self).__init__()

        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes

    def model(self, x, labels=None, kl_factor=1.0, scale=1.0):
        with poutine.scale(scale=scale):
            n_data = x.size(0)  # aka batch size
            a1_mean = torch.zeros(self.input_dim, self.layer_neurons[0])
            a1_scale = torch.ones(self.input_dim, self.layer_neurons[0])
            a1_dropout = torch.tensor(0.25)
            a2_mean = torch.zeros(self.layer_neurons[0] + 1, self.layer_neurons[0])
            a2_scale = torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0])
            a2_dropout = torch.tensor(1.0)
            a3_mean = torch.zeros(self.layer_neurons[0] + 1, self.layer_neurons[0])
            a3_scale = torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0])
            a3_dropout = torch.tensor(1.0)
            a4_mean = torch.zeros(self.layer_neurons[0] + 1, self.num_classes)
            a4_scale = torch.ones(self.layer_neurons[0] + 1, self.num_classes)
            # Mark batched calculations to be conditionally independent given parameters using `plate`
            with pyro.plate('data', size=n_data):
                # Sample first hidden layer
                h1 = pyro.sample('h1', bnn.HiddenLayer(x, a1_mean, a1_dropout * a1_scale,
                                                       non_linearity=nnf.leaky_relu,
                                                       KL_factor=kl_factor))
                # Sample second hidden layer
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                       non_linearity=nnf.leaky_relu,
                                                       KL_factor=kl_factor))
                # Sample third hidden layer
                h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                       non_linearity=nnf.leaky_relu,
                                                       KL_factor=kl_factor))
                # Sample output logits
                logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                               non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                               KL_factor=kl_factor,
                                                               include_hidden_bias=False))
                # One-hot encode labels
                labels = nnf.one_hot(labels) if labels is not None else None
                # Condition on observed labels, so it calculates the log-likehood loss when training using VI
                return pyro.sample('label', dist.OneHotCategorical(logits=logits), obs=labels)

    def guide(self, x, labels=None, kl_factor=1.0, scale=1.0):
        n_data = x.size(0)
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.input_dim, self.layer_neurons[0]))
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.input_dim, self.layer_neurons[0]),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))
        a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.layer_neurons[0] + 1, self.layer_neurons[0]))
        a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0]),
                              constraint=constraints.greater_than(0.01))
        a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.layer_neurons[0] + 1, self.layer_neurons[0]))
        a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0]),
                              constraint=constraints.greater_than(0.01))
        a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.layer_neurons[0] + 1, self.num_classes))
        a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.layer_neurons[0] + 1, self.num_classes),
                              constraint=constraints.greater_than(0.01))
        with pyro.plate('data', size=n_data):
            h1 = pyro.sample('h1', bnn.HiddenLayer(x, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))


    def forward(self, x, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(x)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0)


    def predict(self, x, num_samples):
        with torch.autograd.no_grad():
            return self.forward(x, n_samples=num_samples).detach().cpu().numpy()

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)


class BaseNet3(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(BaseNet3, self).__init__()
        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes
        # TODO add dropout
        self.fc1 = nn.Linear(input_dim, layer_neurons[0])
        self.fc2 = nn.Linear(layer_neurons[0], layer_neurons[1])
        self.fc3 = nn.Linear(layer_neurons[1], layer_neurons[2])
        self.out = nn.Linear(layer_neurons[2], num_classes)
    def forward(self, x):
        # Layer 1
        z1 = self.fc1(x)
        a1 = F.relu(z1)
        # Layer 2
        z2 = self.fc2(a1)
        a2 = F.relu(z2)
        # Layer 3
        z3 = self.fc3(a2)
        a3 = F.relu(z3)
        # Output
        output = self.out(a3)
        return output

class BayesNN3Lift(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(BayesNN3Lift, self).__init__()

        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes

        self.basenet = BaseNet3(input_dim, layer_neurons, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softplus = torch.nn.Softplus()
        from pyro.infer.autoguide import AutoDiagonalNormal
        self.guide = AutoDiagonalNormal(self.model)

    def model(self, x, labels=None, kl_factor=1.0):
        # Layer 1
        fc1w_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.fc1.weight), scale=torch.ones_like(self.basenet.fc1.weight))
        fc1b_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.fc1.bias), scale=torch.ones_like(self.basenet.fc1.bias))

        # Layer 2
        fc2w_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.fc2.weight), scale=torch.ones_like(self.basenet.fc2.weight))
        fc2b_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.fc2.bias), scale=torch.ones_like(self.basenet.fc2.bias))

        # Layer 3
        fc3w_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.fc3.weight), scale=torch.ones_like(self.basenet.fc3.weight))
        fc3b_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.fc3.bias), scale=torch.ones_like(self.basenet.fc3.bias))

        # Output Layer
        outw_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.out.weight), scale=torch.ones_like(self.basenet.out.weight))
        outb_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.basenet.out.bias), scale=torch.ones_like(self.basenet.out.bias))

        priors = {'fc1.weight': fc1w_prior,
                  'fc1.bias': fc1b_prior,
                  'fc2.weight': fc2w_prior,
                  'fc2.bias': fc2b_prior,
                  'fc3.weight': fc3w_prior,
                  'fc3.bias': fc3b_prior,
                  'out.weight': outw_prior,
                  'out.bias': outb_prior}

        lifted_module = pyro.random_module("module",self.basenet, priors)
        # Sample a model
        lifted_reg_model = lifted_module()
        lhat = self.log_softmax(lifted_reg_model(x))
        pyro.sample("obs", Categorical(logits=lhat), obs=labels)


    def guidebad(self, x, labels=None, kl_factor=1.0):
        if x is not None:
            n_data = x.size(0)

        # First layer - weights
        fc1w_mu = torch.randn_like(self.basenet.fc1.weight)
        fc1w_sigma = torch.randn_like(self.basenet.fc1.weight)
        # why wrap with softplus???
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = self.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)

        # First layer - biases
        fc1b_mu = torch.randn_like(self.basenet.fc1.bias)
        fc1b_sigma = torch.randn_like(self.basenet.fc1.bias)
        # why wrap with softplus???
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = self.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # Second layer - weights
        fc2w_mu = torch.randn_like(self.basenet.fc2.weight)
        fc2w_sigma = torch.randn_like(self.basenet.fc2.weight)
        fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
        fc2w_sigma_param = self.softplus(pyro.param("fc2w_sigma", fc2w_sigma))
        fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)

        # Second layer - biases
        fc2b_mu = torch.randn_like(self.basenet.fc2.bias)
        fc2b_sigma = torch.randn_like(self.basenet.fc2.bias)
        # why wrap with softplus???
        fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
        fc2b_sigma_param = self.softplus(pyro.param("fc2b_sigma", fc2b_sigma))
        fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)


        # Third layer - weights
        fc3w_mu = torch.randn_like(self.basenet.fc3.weight)
        fc3w_sigma = torch.randn_like(self.basenet.fc3.weight)
        fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
        fc3w_sigma_param = self.softplus(pyro.param("fc3w_sigma", fc3w_sigma))
        fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)

        # Third layer - biases
        fc3b_mu = torch.randn_like(self.basenet.fc3.bias)
        fc3b_sigma = torch.randn_like(self.basenet.fc3.bias)
        # why wrap with softplus???
        fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
        fc3b_sigma_param = self.softplus(pyro.param("fc3b_sigma", fc3b_sigma))
        fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)


        # Output layer - weights
        outw_mu = torch.randn_like(self.basenet.out.weight)
        outw_sigma = torch.randn_like(self.basenet.out.weight)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = self.softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)


        # Third layer - biases
        outb_mu = torch.randn_like(self.basenet.out.bias)
        outb_sigma = torch.randn_like(self.basenet.out.bias)
        # why wrap with softplus???
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = self.softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

        priors = {'fc1.weight': fc1w_prior,
                  'fc1.bias': fc1b_prior,
                  'fc2.weight': fc2w_prior,
                  'fc2.bias': fc2b_prior,
                  'fc3.weight': fc3w_prior,
                  'fc3.bias': fc3b_prior,
                  'out.weight': outw_prior,
                  'out.bias': outb_prior}

        lifted_module = pyro.random_module("module", self.basenet, priors)

        return lifted_module()

    def forward(self, x, n_samples=10):
        sampled_models = [self.guide(None, None) for _ in range(n_samples)]
        res = [mod(x).data for mod in sampled_models]
        return torch.stack(res, dim=0)


    def predict(self, x, num_samples):
        return self.forward(x, n_samples=num_samples).detach().cpu().numpy()

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)


class NeighbourBayesNN3(nn.Module):
    def __init__(self, input_dim, layer_neurons, num_classes):
        super(NeighbourBayesNN3, self).__init__()

        self.input_dim = input_dim
        self.layer_neurons = layer_neurons
        self.num_classes = num_classes

    def model(self, x, labels=None, kl_factor=1.0):
        n_data = x.size(0) # Batch size
        a1_mean = torch.zeros(self.input_dim, self.layer_neurons[0])
        a1_scale = torch.ones(self.input_dim, self.layer_neurons[0])
        a1_dropout = torch.tensor(0.25)
        a2_mean = torch.zeros(self.layer_neurons[0] + 1, self.layer_neurons[0])
        a2_scale = torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0])
        a2_dropout = torch.tensor(1.0)
        a3_mean = torch.zeros(self.layer_neurons[0] + 1, self.layer_neurons[0])
        a3_scale = torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0])
        a3_dropout = torch.tensor(1.0)
        a4_mean = torch.zeros(self.layer_neurons[0] + 1, self.num_classes)
        a4_scale = torch.ones(self.layer_neurons[0] + 1, self.num_classes)
        with pyro.plate('data', size=n_data):
            # Sample first hidden layer
            h1 = pyro.sample('h1', bnn.HiddenLayer(x, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample second hidden layer
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample third hidden layer
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample output logits
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            # One-hot encode labels
            # labels = nnf.one_hot(labels) if labels is not None else None
            # Condition on observed labels, so it calculates the log-likehood loss when training using VI
            sampled =  pyro.sample('label', dist.Dirichlet(concentration=logits), obs=labels)
            return sampled

    def guide(self, x, labels=None, kl_factor=1.0):
        n_data = x.size(0)
        # Set-up parameters to be optimized to approximate the true posterior
        # Mean parameters are randomly initialized to small values around 0, and scale parameters
        # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
        # the prior scale of 1.
        # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
        # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
        # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.input_dim, self.layer_neurons[0]))
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.input_dim, self.layer_neurons[0]),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))
        a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.layer_neurons[0] + 1, self.layer_neurons[0]))
        a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0]),
                              constraint=constraints.greater_than(0.01))
        a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.layer_neurons[0] + 1, self.layer_neurons[0]))
        a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.layer_neurons[0] + 1, self.layer_neurons[0]),
                              constraint=constraints.greater_than(0.01))
        a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.layer_neurons[0] + 1, self.num_classes))
        a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.layer_neurons[0] + 1, self.num_classes),
                              constraint=constraints.greater_than(0.01))
        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
        with pyro.plate('data', size=n_data):
            h1 = pyro.sample('h1', bnn.HiddenLayer(x, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))


    def forward(self, x, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(x)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0)


    def predict(self, x, num_samples):
        return self.forward(x, n_samples=num_samples).detach().cpu().numpy()

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)

    def dump_params(self, file_path):
        params = {
                "input_dim": self.input_dim,
                "layer_neurons": self.layer_neurons,
                "num_classes": self.num_classes
        }
        json.dump(params, open(file_path, 'w'), indent=4, sort_keys=True)
