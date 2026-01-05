"""
Activation function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSort(nn.Module):
    """
    The GroupSort activation function.

    This layer splits the input features into groups of size `group_size` and sorts 
    the elements within each group in descending order.

    Parameters
    ----------
    group_size : int, optional
        The number of neurons in each group to be sorted, by default 2.
        If the input feature dimension is not divisible by `group_size`, 
        zero-padding is applied internally during the forward pass.
    """
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        """
        Performs the forward pass of the GroupSort activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, features)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, features)`.
            The output has the same shape as the input, with values locally sorted.
        """
        # x shape: (batch_size, features)
        batch_size, features = x.shape

        # pad if necessary to make divisible by group_sort
        remainder = features % self.group_size
        if remainder != 0:
            padding = self.group_size - remainder
            x = F.pad(x, (0, padding))
            features += padding

        # reshape to (batch_size, num_groups, group_size)
        num_groups = features // self.group_size
        x_grouped = x.view(batch_size, num_groups, self.group_size)

        # sort each groups in descending order
        x_sorted, _ = torch.sort(x_grouped, dim=2, descending=True)

        result = x_sorted.view(batch_size, features)

        # remove padding if it was added
        if remainder != 0:
            result = result[:, :-padding]

        return result
