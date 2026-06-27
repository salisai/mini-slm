"""
Standard FFN:  FFN(x) = ReLU(x @ W1) @ W2
SwiGLU FFN:    FFN(x) = (Swish(x @ W1) * (x @ V)) @ W2

The extra matrix V acts as a GATE. For every hidden unit, the model
learns a separate "should this unit be on or off, and how much"
signal (x @ V), and multiplies it elementwise against the main
activation (Swish(x @ W1)). This gives the network finer control
over information flow than a plain activation function alone.

Swish(x) = x * sigmoid(x) — a smooth curve, similar in shape to GeLU.

Because we added an extra weight matrix (V), a gated FFN at the
SAME hidden dimension as a non-gated FFN would have MORE parameters.
To keep parameter count roughly equal, the convention (used by
LLaMA, PaLM, Mistral, etc.) is to shrink the hidden dimension to
8/3 * d_model instead of the standard 4 * d_model for ungated FFNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        """
        Args:
            d_model: model dimension (input and output size of the FFN)
            d_ff: hidden dimension inside the FFN. If None, defaults to
                  the standard GLU convention: round(8/3 * d_model),
                  rounded to a multiple of 8 for clean GPU tensor shapes
                  (most hardware likes dimensions divisible by 8 or 64).
        """
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # round up to nearest multiple of 8 for clean tensor shapes
            d_ff = ((d_ff + 7) // 8) * 8

        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff, bias=False)   
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  
        self.w2 = nn.Linear(d_ff, d_model, bias=False) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., d_model) — works for any leading dimensions

        Returns:
            shape (..., d_model) — same shape as input
        """
        main_branch = F.silu(self.w1(x))     
        gate_branch = self.w3(x)              

        gated = main_branch * gate_branch    

        return self.w2(gated)               

