"""
The complete model: token embedding -> N transformer blocks -> final norm
-> output projection (tied to the embedding weights) -> logits.
"""

import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from block import TransformerBlock


class miniSLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        n_kv_heads: int = 2,
        d_ff: int | None = None,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
    ):
        """
        Args:
            vocab_size:  size of your tokenizer's vocabulary
            d_model:     model hidden dimension
            n_layers:    number of stacked transformer blocks
            n_heads:     number of query attention heads (per block)
            n_kv_heads:  number of key/value attention heads (GQA, per block)
            d_ff:        FFN hidden dim (None = auto 8/3 ratio)
            max_seq_len: longest sequence length you'll ever train/infer on
            rope_base:   RoPE frequency base
            norm_eps:    RMSNorm epsilon
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        #stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                d_ff=d_ff,
                rope_base=rope_base,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.tie_weights()

        # Standard small-init for stability, matching common practice
        # for training transformers from scratch (helps early training
        # stability)
        self.apply(self._init_weights)

    def tie_weights(self):
        self.output_proj.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: shape (batch, seq_len), dtype long,integer
                       token ids from your tokenizer.

        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """

        batch, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )

        x = self.token_embedding(token_ids) # (batch, seq_len, d_model)

        for block in self.blocks:
            x = block(x)                       

        x = self.final_norm(x)                 
        logits = self.output_proj(x) # (batch, seq_len, vocab_size)

        return logits


    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0):
        """
        Simple autoregressive generation: repeatedly predict the next
        token, append it, and feed the whole sequence back in.

        This is intentionally the simplest possible version (no KV
        cache, no top-k/top-p sampling yet), just enough to prove the
        model can produce tokens end to end. We'll add proper sampling
        and a KV cache later in generate.py once training is working.

        Args:
            token_ids: shape (batch, seq_len)

        Returns:
            shape (batch, seq_len + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            input_ids = token_ids[:, -self.max_seq_len:]

            logits = self.forward(input_ids)         
            next_token_logits = logits[:, -1, :]      

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids

