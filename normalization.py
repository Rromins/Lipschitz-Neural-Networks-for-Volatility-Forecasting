"""
Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormLinear(nn.Module):
    """
    Perform Spectral Normalization using power iteration
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
        """Compute largest singular value using power iteration"""
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
        """forward pass"""
        if self.training:
            sigma = self._power_iteration(n_iterations=None)
        else:
            with torch.no_grad():
                sigma = self._power_iteration(n_iterations=None)

        # normalize weights
        weight = self.lipschitz_const * self.weight / sigma

        return F.linear(x, weight, self.bias)
