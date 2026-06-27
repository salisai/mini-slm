"""
Grouped Query Attention (GQA): a middle ground. Queries are split
into GROUPS. All query heads within a group share one key head and
one value head. More KV heads than MQA (better quality), fewer than
full multi-head (better inference efficiency).

If n_kv_heads == n_heads, this becomes standard multi-head attention.
If n_kv_heads == 1, this becomes MQA.

RoPE is applied to Q and K (never V) right after projecting them,
before the attention dot product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rope import RotaryEmbedding, apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        """
        Args:
            d_model:     total model dimension (e.g. 384)
            n_heads:     number of QUERY heads
            n_kv_heads:  each group has n_heads/n_kv_heads
                         query heads sharing one kv head.
            max_seq_len: longest sequence RoPE will precompute angles for
            rope_base:   RoPE frequency base (10000 is standard)
        """

        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads # number of query heads that will share a single KV head


        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
            causal_mask: if True, each token can only attend to itself
                         and earlier tokens.

        Returns:
            shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)


        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # q shape: (batch, n_heads, seq_len, head_dim)

        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k, v shape: (batch, n_kv_heads, seq_len, head_dim)

        cos, sin = self.rope(seq_len) # compute an array of rotation angles for this seq_len
        q = apply_rope(q, cos, sin) # 
        k = apply_rope(k, cos, sin)

 
        k = self._repeat_kv(k)  # (batch, n_heads, seq_len, head_dim)
        v = self._repeat_kv(v)  # (batch, n_heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: (batch, n_heads, seq_len, seq_len)

        if causal_mask:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
            scores = scores + mask.to(scores.device)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # merge heads back together and project to d_model
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeats each kv head n_rep times along the head dimension so
        the kv tensor lines up with the number of query heads.

        Input:  (batch, n_kv_heads, seq_len, head_dim)
        Output: (batch, n_kv_heads * n_rep, seq_len, head_dim) == (batch, n_heads, ...)
        """
        if self.n_rep == 1:
            return x  # no repetition needed (this is standard MHA)

        batch, n_kv_heads, seq_len, head_dim = x.shape

        x = x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch, n_kv_heads * self.n_rep, seq_len, head_dim)

 