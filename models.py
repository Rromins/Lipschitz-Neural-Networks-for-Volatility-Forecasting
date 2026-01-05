"""
Neural Networks models
"""
from normalization import SpectralNormLinear
from activations import GroupSort
import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    """
    A simple feedforward neural network (MLP) designed for regression tasks.

    This network consists of three fully connected (linear) layers with ReLU 
    activations in between. It reduces the input dimensionality progressively 
    down to a single scalar output.

    Architecture
    ------------
    1. Input Layer:  (Batch_Size, in_features)
    2. Hidden 1:     Linear -> (Batch_Size, 64) + ReLU
    3. Hidden 2:     Linear -> (Batch_Size, 32) + ReLU
    4. Output Layer: Linear -> (Batch_Size, 1)

    Parameters
    ----------
    in_features : int
        The number of input features (dimensionality of the input vector).

    Attributes
    ----------
    layer1 : nn.Linear
        First dense layer (Input -> 64).
    layer2 : nn.Linear
        Second dense layer (64 -> 32).
    layer3 : nn.Linear
        Output dense layer (32 -> 1).
    relu : nn.ReLU
        Rectified Linear Unit activation function.
    """
    def __init__(self, in_features):
        super(FeedforwardNN, self).__init__()
        self.layer1 = nn.Linear(in_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, in_features)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, 1)`.
            Contains the predicted regression values.
        """
        value = self.layer1(x)
        value = self.relu(value)
        value = self.layer2(value)
        value = self.relu(value)
        value = self.layer3(value)
        return value


class LipschitzNN(nn.Module):
    """
    A Lipschitz-constrained Multi-Layer Perceptron (MLP).

    This network enforces a global Lipschitz constant on the mapping by combining 
    Spectrally Normalized Linear layers with GroupSort activation functions. 

    Architecture
    ------------
    The network is constructed dynamically based on `hidden_dims`:
    Input -> [SpectralLinear -> GroupSort] x N -> SpectralLinear -> Output

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : list of int
        A list of integers defining the size of each hidden layer.
        Example: `[64, 64]` creates two hidden layers with 64 neurons each.
    output_dim : int
        Dimensionality of the output (e.g., 1 for regression).
    lipschitz_const : float, optional
        The target Lipschitz constant for each linear layer, by default 1.0.
        The global Lipschitz constant is bounded by the product of layer constants.
    nb_iterations : int, optional
        Number of power iterations used to estimate the spectral norm 
        in `SpectralNormLinear` layers, by default 1.
    group_size : int, optional
        The grouping size for the `GroupSort` activation, by default 2.
        GroupSort is a gradient-norm-preserving activation function.

    Attributes
    ----------
    network : nn.Sequential
        The container holding the sequence of layers.
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
        """
        Performs the forward pass of the Lipschitz network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_dim)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, output_dim)`.
        """
        return self.network(x)
