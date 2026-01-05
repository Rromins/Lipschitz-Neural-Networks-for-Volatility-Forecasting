"""
Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormLinear(nn.Module):
    """
    A Linear layer constrained by Spectral Normalization.

    This layer enforces a Lipschitz constant on the linear transformation by 
    rescaling the weight matrix by its largest singular value (spectral norm).
    The spectral norm is approximated efficiently using the Power Iteration method.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    lipschitz_const : float, optional
        The target Lipschitz constant (scaling factor) for the layer.
        Default is 1.0.
    nb_iterations : int, optional
        The number of power iterations to perform per forward pass to estimate 
        the singular value. Default is 1.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The learnable weights of the module of shape `(out_features, in_features)`.
    bias : torch.nn.Parameter
        The learnable bias of the module of shape `(out_features)`.
    u : torch.Tensor (Buffer)
        The left singular vector approximation (persistence buffer).
    v : torch.Tensor (Buffer)
        The right singular vector approximation (persistence buffer).
    """
    def __init__(self,
                 in_features,
                 out_features,
                 lipschitz_const=1.0,
                 nb_iterations=1):
        super().__init__()
        # initialize weight matrix and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.lipschitz_const = lipschitz_const
        self.nb_iterations = nb_iterations

        # initialize power iteration vector
        self.register_buffer('v', torch.randn(in_features))
        self.register_buffer('u', torch.randn(out_features))

    def _power_iteration(self, n_iterations=None):
        """
        Approximates the largest singular value sigma using Power Iteration.

        Updates the internal buffers `u` and `v` to approximate the principal 
        singular vectors of the weight matrix.

        Parameters
        ----------
        n_iterations : int, optional
            Number of iterations to run. If None, uses `self.nb_iterations`.

        Returns
        -------
        torch.Tensor
            The approximated spectral norm (largest singular value) of the weights.
        """
        if n_iterations is None:
            n_iterations = self.nb_iterations
        
        # use detached versions to avoid gradient tracking
        with torch.no_grad():
            v = self.v.clone()
            u = self.u.clone()

            for _ in range(n_iterations):
                u = F.normalize(self.weight @ v, dim=0)
                v = F.normalize(self.weight.t() @ u, dim=0)
            
            # update buffers
            self.v.copy_(v)
            self.u.copy_(u)
        
        # compute singular value
        sigma = torch.dot(u, self.weight @ v)

        return sigma
    
    def forward(self, x):
        """
        Performs the forward pass of the SpectralNormLinear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, in_features)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, out_features)`.
        """
        if self.training:
            sigma = self._power_iteration(n_iterations=None)
        else:
            with torch.no_grad():
                sigma = self._power_iteration(n_iterations=None)

        # normalize weights
        weight = self.lipschitz_const * self.weight / sigma

        return F.linear(x, weight, self.bias)
