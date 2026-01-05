"""
Loss functions
"""

import torch
import torch.nn as nn

class QLIKELoss(nn.Module):
    """
    QLIKE (Quasi-Likelihood) Loss for Volatility Forecasting.
    
    The QLIKE loss function is:
        L(y, h) = y/h - log(y/h) - 1
    
    Where:
        y = true realized volatility
        h = predicted volatility
    
    This loss is commonly used in volatility forecasting as it's robust
    to outliers and doesn't require normality assumptions.
    
    Parameters
    ----------
    eps : float
        Small constant for numerical stability (default: 1e-8)
    reduction : str
        Specifies the reduction to apply: 'mean', 'sum', 'none'
    enforce_positive : bool
        If True, uses softplus to ensure positive predictions
    """
    def __init__(self, eps=1e-8, reduction='mean', enforce_positive=True):
        super(QLIKELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.enforce_positive = enforce_positive

    def forward(self, y_pred, y_true):
        """
        Compute QLIKE loss.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted volatility
        y_true : torch.Tensor
            True realized volatility
            
        Returns
        -------
        loss : torch.Tensor
            QLIKE loss value
        """
        # ensure inputs are positive
        if self.enforce_positive:
            # Use softplus for smooth, differentiable enforcement of positivity
            # softplus(x) = log(1 + exp(x)), always positive and smooth
            y_pred = torch.nn.functional.softplus(y_pred) + self.eps
        else:
            # just add epsilon for numerical stability
            y_pred = y_pred + self.eps

        # True values should already be positive
        y_true = torch.abs(y_true) + self.eps

        # compute QLIKE: y/h - log(y/h) - 1
        ratio = y_true / y_pred
        loss = ratio - torch.log(ratio) - 1

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss

