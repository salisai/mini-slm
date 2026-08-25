"""
Training loop for miniSLM: load corpus -> tokenize -> causal LM loss -> checkpoints.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))

from model import miniSLM  # noqa: E402
from tokenizer.train_tokenizer import load_tokenizer  # noqa: E402


@dataclass
class TrainConfig:
    model: dict
    tokenizer: dict
    training: dict


def load_config(path: str | Path = "config.yaml") -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(
        model=raw["model"],
        tokenizer=raw["tokenizer"],
        training=raw["training"],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TokenChunkDataset(Dataset):
    """Non-overlapping fixed-length windows from a 1-D token stream."""

    def __init__(self, token_ids: list[int], seq_len: int):
        if len(token_ids) <= seq_len:
            raise ValueError(
                f"Need more than {seq_len} tokens for training, got {len(token_ids)}."
            )
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.n_chunks = (len(token_ids) - seq_len) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return x, y


def tokenize_corpus(path: str | Path, tokenizer) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    return tokenizer.encode(text).ids


def split_token_stream(
    token_ids: list[int], val_split: float
) -> tuple[list[int], list[int]]:
    split_idx = int(len(token_ids) * (1.0 - val_split))
    return token_ids[:split_idx], token_ids[split_idx:]


def build_model(cfg: TrainConfig, vocab_size: int, device: torch.device) -> miniSLM:
    model = miniSLM(vocab_size=vocab_size, **cfg.model)
    return model.to(device)


@torch.no_grad()
def evaluate(model: miniSLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    model.train()
    return total_loss / max(total_tokens, 1)


def lr_at_step(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return 0.0
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    model: miniSLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    val_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "model": cfg.model,
                "tokenizer": cfg.tokenizer,
            },
            "val_loss": val_loss,
        },
        path,
    )


def train(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    tcfg = cfg.training
    set_seed(tcfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    tokenizer_path = ROOT / cfg.tokenizer["path"]
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run tokenizer/train_tokenizer.py first."
        )

    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    data_path = ROOT / tcfg["data_path"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Add your corpus file before training."
        )

    print(f"Tokenizing corpus: {data_path}")
    all_tokens = tokenize_corpus(data_path, tokenizer)
    train_tokens, val_tokens = split_token_stream(all_tokens, tcfg["val_split"])
    print(
        f"Tokens: {len(all_tokens):,} total | "
        f"{len(train_tokens):,} train | {len(val_tokens):,} val"
    )

    seq_len = cfg.model["max_seq_len"]
    train_ds = TokenChunkDataset(train_tokens, seq_len)

    val_loader: DataLoader | None = None
    if len(val_tokens) > seq_len:
        val_ds = TokenChunkDataset(val_tokens, seq_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=tcfg["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        val_chunks = len(val_ds)
    else:
        val_chunks = 0
        print(
            f"Validation split has only {len(val_tokens)} tokens "
            f"(need > {seq_len}); skipping val evaluation."
        )

    print(f"Chunks: {len(train_ds):,} train | {val_chunks:,} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(tcfg["batch_size"], len(train_ds)),
        shuffle=True,
        drop_last=len(train_ds) >= tcfg["batch_size"],
    )

    model = build_model(cfg, vocab_size, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
    )

    checkpoint_dir = ROOT / tcfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = tcfg.get("use_wandb", False)
    if use_wandb:
        import wandb

        wandb.init(project=tcfg.get("wandb_project", "mini-slm"), config=yaml.safe_load(open(config_path)))

    model.train()
    step = 0
    max_steps = tcfg["max_steps"]
    running_loss = 0.0
    loader_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x = x.to(device)
        y = y.to(device)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_at_step(step, tcfg["learning_rate"], tcfg["warmup_steps"], max_steps)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        running_loss += loss.item()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if step % tcfg["log_interval"] == 0:
            avg_loss = running_loss / tcfg["log_interval"]
            if use_wandb:
                wandb.log({"train/loss": avg_loss, "train/lr": optimizer.param_groups[0]["lr"]}, step=step)
            running_loss = 0.0

        if val_loader is not None and step % tcfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"\nStep {step}: val_loss={val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)

        if step % tcfg["save_interval"] == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            val_loss_snapshot = (
                evaluate(model, val_loader, device) if val_loader is not None else None
            )
            save_checkpoint(ckpt_path, model, optimizer, step, cfg, val_loss_snapshot)
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, cfg, val_loss_snapshot)
            print(f"Saved checkpoint to {ckpt_path}")

    pbar.close()

    final_val_loss = evaluate(model, val_loader, device) if val_loader is not None else None
    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, cfg, final_val_loss)
    if final_val_loss is not None:
        print(f"Training complete. Final val_loss={final_val_loss:.4f}")
    else:
        print("Training complete.")
    print(f"Latest checkpoint: {checkpoint_dir / 'latest.pt'}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
