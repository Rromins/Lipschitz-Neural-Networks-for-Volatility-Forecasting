"""
Neural Networks models
"""
from normalization import SpectralNormLinear
from activations import GroupSort
import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    """
    A simple feedforward neural network for regression tasks.

    Architecture
    ------------
        Input (in_features)
        Linear(in_features to 64)
            ReLU
        Linear(64 to 32)
            ReLU
        Linear(32 to 1)
        Output (predicted value)
    """
    def __init__(self, in_features):
        super(FeedforwardNN, self).__init__()
        self.layer1 = nn.Linear(in_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Define the forward pass of the network
        """
        value = self.layer1(x)
        value = self.relu(value)
        value = self.layer2(value)
        value = self.relu(value)
        value = self.layer3(value)
        return value


class LipschitzNN(nn.Module):
    """
        Lipschitz-constrained MLP with GroupSort activation.
        
        Parameters
        ----------
        input_dim: int
            Input dimension
        hidden_dims: list of int
            Hidden layer dimensions (e.g., [64, 64])
        output_dim: int
            Output dimension
        lipschitz_const: float
            Lipschitz constant for the entire network
        nb_iterations: int
            Number of iterations for the power iteration algorithm
        group_size: int
            Group size for GroupSort (typically 2)
        """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 lipschitz_const=1.0,
                 nb_iterations=1,
                 group_size=2):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dim + [output_dim]

        for i in range(len(dims)-1):
            layers.append(SpectralNormLinear(dims[i],
                                             dims[i+1],
                                             lipschitz_const=lipschitz_const,
                                             nb_iterations=nb_iterations))

            if i < len(dims) - 2: # no activation function on the output layer 
                layers.append(GroupSort(group_size=group_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)
