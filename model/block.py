"""
Pre-norm structure:
    x_out = x + Attention(RMSNorm(x))
    x_out = x_out + FFN(RMSNorm(x_out))

The KEY property: the residual highway (the "+ x" part) NEVER
passes through RMSNorm. Only the input to the side-branch
(Attention or FFN) gets normalized. This is exactly what gives us:
    d(x_out)/d(x_in) = 1 + branch_gradient
guaranteeing gradient flow of magnitude >= ~1 through every layer,
regardless of how deep the network is.

This file stacks TWO of these sub-blocks together:
1. norm -> attention -> add back to residual highway
2. norm -> FFN       -> add back to residual highway
"""

import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from attention import GroupedQueryAttention
from ffn import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        d_ff: int | None = None,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
    ):
        """
        Args:
            d_model:     model dimension
            n_heads:     number of query attention heads
            n_kv_heads:  number of key/value attention heads (GQA)
            max_seq_len: longest sequence RoPE will precompute for
            d_ff:        FFN hidden dimension (None = auto 8/3 ratio)
            rope_base:   RoPE frequency base
            norm_eps:    epsilon for RMSNorm numerical stability
        returns: 
            shape (batch, seq_len, d_model) same shape as input
        """
        super().__init__()

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )

        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)

        Returns:
            shape (batch, seq_len, d_model) — same shape as input
        """

        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))

        return x

