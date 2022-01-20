"""Actors for gym environments."""
import numpy as np
import torch
from torch import nn


class MLPActor(nn.Module):
    """MLP policy for continuous control.

    Feedforward network with identical activations on every layer. Takes in a
    continuous observation and outputs a continuous action normalized to the
    range (-1, 1) with tanh.

    Some methods return self so that you can do actor = Actor().method()

    Args:
        layer_shapes: List of tuples of (in_shape, out_shape) for linear layers.
        activation: Activation layer class, e.g. nn.Tanh
    """

    def __init__(self, layer_shapes, activation):
        super().__init__()

        layers = []
        for i, shape in enumerate(layer_shapes):
            layers.append(nn.Linear(*shape))
            if i == len(layer_shapes) - 1:
                # tanh on last layer.
                layers.append(nn.Tanh())
            else:
                layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, obs):
        """Computes actions for a batch of observations."""
        return self.model(obs)

    def action(self, obs):
        """Computes action for one observation."""
        obs = torch.from_numpy(obs[None].astype(np.float32))
        return self(obs)[0].cpu().detach().numpy()

    def initialize(self, func):
        """Initializes weights for Linear layers with func.

        func usually comes from nn.init
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                func(m.weight)
                nn.init.zeros_(m.bias)  # Biases init to zero.

        self.apply(init_weights)

        return self

    def serialize(self):
        """Returns 1D array with all parameters in the actor."""
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.model.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self

    def gradient(self):
        """Returns 1D array with gradient of all parameters in the actor."""
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()])
