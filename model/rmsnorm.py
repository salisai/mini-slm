"""
LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
RMSNorm:    y = x / sqrt(mean(x^2) + eps) * gamma
 
RMSNorm does NOT subtract the mean, and has NO bias (beta).
It only rescales the vector by its root-mean-square magnitude, then applies
a single learnable scale (gamma). Fewer operations, fewer parameters to
move from memory -> faster, and works just as well in practice. More saving at runtime.
"""

import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: the size of the last dimension of the input (d_model).
                 RMSNorm normalizes ACROSS this dimension, independently
                 for every token, every batch element.
            eps: tiny constant added inside the square root to prevent
                 division by zero if a vector happens to be all zeros.
        """
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim)) # trainable, gamma 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., dim) — works for any leading
               dimensions, eg. (batch, sequence length, dim), normalization
               always happens over the LAST dimension only. BatchNorm normalizes 
               over the first dimension, batch. 
 
        Returns:
            normalized tensor, same shape as input.
        """
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_sq + self.eps)
        return x_normed * self.weight
    