import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import math
from habitat_modelling.ml.torch.models.blocks import Dense


class BinaryGPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(BinaryGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class MulticlassGPClassificationModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        self.num_dim = num_dim
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_size=num_dim
        )
        variational_strategy = gpytorch.variational.AdditiveGridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=[grid_bounds], num_dim=num_dim,
            variational_distribution=variational_distribution, mixing_params=False, sum_output=False
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds
        self.grid_size = grid_size



    def forward(self, x):
        x = gpytorch.utils.grid.scale_to_bounds(x, self.grid_bounds[0], self.grid_bounds[1])
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def dump_params(self, path):
        params = {
            "num_dim": self.num_dim,
            "grid_bounds": self.grid_bounds,
            "grid_size": self.grid_size
        }
        json.dump(params, open(path, 'w'))

class GPClassifierInference:
    def __init__(self, gpc, likelihood):
        self.gpc = gpc
        self.likelihood = likelihood
        # Put into eval mode
        self.gpc.eval()
        self.likelihood.eval()

    def predict(self, x, num_samples=10):
        x[x < self.gpc.grid_bounds[0]] = self.gpc.grid_bounds[0]
        x[x > self.gpc.grid_bounds[1]] = self.gpc.grid_bounds[1]
        with torch.no_grad(),  gpytorch.settings.num_likelihood_samples(num_samples):
            out = self.likelihood(self.gpc(x))
        return out.probs.detach().cpu().numpy()





class GaussianProcessLayer(gpytorch.models.AbstractVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_size=num_dim
        )
        variational_strategy = gpytorch.variational.AdditiveGridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=[grid_bounds], num_dim=num_dim,
            variational_distribution=variational_distribution, mixing_params=False, sum_output=False
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res


class DKLDense(gpytorch.Module):
    def __init__(self, layer_neurons, num_dim, grid_bounds=(-10., 10.)):
        super(DKLDense, self).__init__()
        layers = []
        self.num_dim = num_dim
        self.grid_bounds = grid_bounds
        self.layer_neurons = layer_neurons

        activation = nn.ReLU()

        in_units = num_dim
        for n, neurons in enumerate(layer_neurons):
            out_units = neurons
            layers.append(Dense(in_units, out_units, activation, dropout=0.1))
            in_units = out_units

        self.feature_extractor = nn.Sequential(*layers)

        self.dkl = DKLModel(self.feature_extractor, out_units, grid_bounds=grid_bounds)

    def forward(self, x):
        return self.dkl(x)

    def dump_params(self, path):
        params = {
            "layer_neurons": self.layer_neurons,
            "num_dim": self.num_dim,
            "grid_bounds": self.grid_bounds
        }
        json.dump(params, open(path, 'w'))


class GPSKWrapper:
    def __init__(self, model):
        self.latent_model = model
    def predict(self, X, num_samples):
        """
        Args:
            X:

        Returns:
        """
        #TODO find how to get
        x = X.detach().cpu().numpy()
        preds = []
        for n in range(num_samples):
            preds.append(self.latent_model.predict_proba(x))
        return np.concatenate(preds, axis=0)


