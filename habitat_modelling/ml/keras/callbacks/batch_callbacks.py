# Batch Callbacks
#   Methods to be used when train_on_batch is being used. Doesn't subclass keras callback.

import keras
import numpy as np
import keras.backend as K
import tensorflow as tf


def calculate_next_lr_step(epoch, lr, step_multiplier=0.1, step_rate=10):
    """

    Args:
        epoch: (int) current epoch
        lr: (float) the initial learning rate
        step_multiplier: (float) the adjustment to the current learning rate per step (e.g. 0.2)
        step_rate: (int) step

    Returns:
        (float): Learning Rate
    """
    if epoch > 0 and epoch % step_rate == 0:
        lr = lr*step_multiplier
    return lr


def step_learning_rate(kvar, epoch, step_multiplier, step_rate):
    """
    Applies a proportional step learning rate.

    Usage:
        step_learning_rate(model.optimizer.lr, epoch=10, step_multiplier=0.1, step_rate=5)  # Works in place

    Args:
        kvar: (Keras Variable) The keras optimizer lr variable
        epoch: (int) The epoch number
        step_multiplier: (float) the amount to proportionally reduce by each epoch
        step_rate: (int) the interval at which to decrease by.

    Returns:
        None
    """
    curr_lr = K.get_value(kvar)
    curr_lr = calculate_next_lr_step(epoch, curr_lr, step_multiplier, step_rate)
    K.set_value(kvar, curr_lr)


def linear_learning_rate(kvar, initial_lr, rate, limit):
    """
    Applies a linear learning rate update.

    Usage:
        linear_learning_rate(model.optimizer.lr, initial_lr=1e-3, rate=-0.001,  limit=1e-7)  # Works in place

    Args:
        kvar: (Keras Variable) The keras optimizer lr variable
        initial_lr: (float) the initial learning rate
        rate: (float) the learning rate decay. Should be negative!
        limit: (float) the minimum learning rate.

    Returns:
        None
    """
    lr = initial_lr - rate*initial_lr
    lr = max(lr, limit)
    K.set_value(kvar, lr)


def loss_weights_updater(loss_weights_k, losses, target_ratio, cap):
    """
    This updates the loss weights such that they are weighted according to a set ratio.
    Useful when there are competing loss functions - e.g. kl and mse.

    Usage:
        # Importing
        import keras.backend as K
        # When compiling
        loss_weights = K.variable([0.1, 0.5])
        model.compile(optimizer=optimizer, loss=['mse','binary_cross_entropy'], loss_weights=loss_weights)
        # When updating (e.g. at end of epoch)
        loss_weights_updater(loss_weights, np.array([1.0, 0.5]), cap=np.array([100.0, 100.0])
        linear_learning_rate(model.optimizer.lr, initial_lr=1e-3, rate=-0.001,  limit=1e-7)  # Works in place

    Args:
        loss_weights: (Keras Variable) the current loss weights
        losses: (np.ndarray) the current losses
        target_ratio: (np.ndarray) the target ratio between the weights
        cap: (np.ndarray) the maximum value of each loss weight

    Caveats:
        Can't use model.save - use model.save_weights instead

    Returns:
        None
    """
    lw = K.get_value(loss_weights_k)
    if len(lw) != len(target_ratio) or len(lw) != len(losses) or len(lw) != cap:
        raise ValueError("Arrays have to be of same length")
    weights = target_ratio/losses
    weights = np.minimum(weights, cap)
    K.set_value(loss_weights_k, weights)
    # TODO this won't work yet - needs improvements from Keras regarding saving models
    # Can be used if model.save_weights is used instead of model.save


def learning_rate_balancer(lr_tensors, base_lr, losses, cap):
    """
    This balances the learning rates of optimizers so they are set according to a ratio. Useful in training GANs.

    Args:
        lr_tensors: (list of Tensors) A list of lr_tensors to balance - get from optimizers. e.g. model.optimizer.lr
        base_lr: The base learning rate to set to
        losses: (np.ndarray) The losses to base the balancing on
        cap: (list) A multiplier cap on the LRs.
    Returns:
        None

    """
    # lr_vals = np.array([K.get_value(t) for t in lr_tensors])
    weights = losses/np.sum(losses)
    new_lrs = base_lr*weights
    for t,lr in zip(lr_tensors, new_lrs):
        K.set_value(t, lr)








