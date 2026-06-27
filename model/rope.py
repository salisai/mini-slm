"""
Recap from the lecture:
    A good relative position embedding should satisfy:
        <f(x, i), f(y, j)> = g(x, y, i - j)
    Meaning: the inner product (dot product) of two embedded vectors
    should depend ONLY on their relative distance (i - j), not their
    absolute positions i and j.

How it's implemented:
    1. Take a vector of dimension `head_dim`.
    2. Split it into pairs of dimensions: (x0,x1), (x2,x3), (x4,x5), ...
    3. Rotate EACH PAIR independently in its own little 2D plane.
    4. Different pairs rotate at different SPEEDS (frequencies) — this
       is exactly like how sine/cosine embeddings used multiple
       frequencies, except here it's a rotation, not an addition.

This is applied INSIDE every attention layer, directly to queries and
keys, right before the attention dot product — NOT once at the input
embedding layer. That's what makes it "relative" rather than "absolute."
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Precomputational cache. Precomputes the rotation angles (as cos/sin values) for every
    position up to some maximum sequence length, for a given head_dim.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            head_dim: dimension of EACH attention head (NOT d_model).
                      Must be even, since we rotate pairs of dimensions.
            max_seq_len: longest sequence length we precompute angles for.
            base: controls the range of rotation frequencies. 10000 is
                  the standard value used in the original RoPE paper
                  and essentially every model since (LLaMA, GPT-J, etc).
        """

        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even: RoPE rotates pairs of dimensions"

        self.head_dim = head_dim
        # Formula: theta_k = base^(-2k/head_dim) for k = 0, 1, ..., head_dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)


    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len).float()  # shape: (seq_len,)

        # shape: (seq_len, head_dim/2)
        angles = torch.outer(positions, self.inv_freq)

        # Precompute cos and sin of every angle. These are what actually
        # get used to rotate query/key vectors.
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)


    def forward(self, seq_len: int):
        """
        Returns the cos/sin values needed for a sequence of this length.

        Returns:
            cos: shape (seq_len, head_dim/2)
            sin: shape (seq_len, head_dim/2)
        """
        if seq_len > self.cos_cache.shape[0]:
            # Sequence longer than what we precomputed — rebuild cache bigger.
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]



def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies the actual rotation to a query or key tensor.

    Args:
        x:   shape (batch, n_heads, seq_len, head_dim) — query or key tensor
        cos: shape (seq_len, head_dim/2) — from RotaryEmbedding.forward()
        sin: shape (seq_len, head_dim/2) — from RotaryEmbedding.forward()

    Returns:
        rotated tensor, same shape as x.
    """
    x1, x2 = x.chunk(2, dim=-1) 

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return torch.cat([rotated_x1, rotated_x2], dim=-1)


