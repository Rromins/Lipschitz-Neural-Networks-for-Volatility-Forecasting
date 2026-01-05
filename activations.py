"""
Activation function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSort(nn.Module):
    """
    GroupSort activation function.
    
    Groups neurons into pairs and sorts each pair in descending order.
    This activation has Lipschitz constant = 1.
    
    Parameters
    ----------
    group_size : int
        Size of each group (typically 2)
    """
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        """forward"""
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
