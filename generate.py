"""
Text generation for miniSLM: load checkpoint + tokenizer, sample with temperature / top-k / top-p.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from model import miniSLM  # noqa: E402
from train_tokenizer import load_tokenizer  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[miniSLM, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    tokenizer_cfg = config["tokenizer"]
    model_cfg = config["model"]

    tokenizer_path = ROOT / tokenizer_cfg["path"]
    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    model = miniSLM(vocab_size=vocab_size, **model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, {
        "tokenizer": tokenizer,
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
    }


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample one token from logits of shape (batch, vocab)."""
    logits = logits.clone()

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, next_token)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_text(
    model: miniSLM,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    device: torch.device | None = None,
) -> str:
    device = device or get_device()
    token_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = token_ids[:, -model.max_seq_len :]
        logits = model(context)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token_ids = torch.cat([token_ids, next_token], dim=1)

        eot_id = tokenizer.token_to_id("<|endoftext|>")
        if eot_id is not None and next_token.item() == eot_id:
            break

    return tokenizer.decode(token_ids[0].tolist())


def load_generation_defaults(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw.get("generation", {})


def generate(
    prompt: str,
    *,
    checkpoint: str | None = None,
    config_path: str = "config.yaml",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    defaults = load_generation_defaults(config_path)
    checkpoint = checkpoint or defaults.get("checkpoint", "checkpoints/latest.pt")
    max_new_tokens = max_new_tokens if max_new_tokens is not None else defaults.get("max_new_tokens", 128)
    temperature = temperature if temperature is not None else defaults.get("temperature", 0.8)
    top_k = top_k if top_k is not None else defaults.get("top_k", 50)
    top_p = top_p if top_p is not None else defaults.get("top_p", 0.9)

    device = get_device()
    ckpt_path = ROOT / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Train the model first with `python main.py train`."
        )

    model, meta = load_checkpoint(ckpt_path, device)
    step = meta.get("step")
    val_loss = meta.get("val_loss")
    print(f"Loaded checkpoint (step={step}, val_loss={val_loss})")

    output = generate_text(
        model,
        meta["tokenizer"],
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with miniSLM")
    parser.add_argument("prompt", nargs="?", default="Once upon a time", help="Prompt text")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = generate(
        args.prompt,
        checkpoint=args.checkpoint,
        config_path=args.config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)
