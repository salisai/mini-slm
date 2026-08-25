"""
CLI entry point for mini-SLM.

Examples:
    python main.py train
    python main.py train --config config.yaml
    python main.py generate "The transformer architecture"
    python main.py generate --checkpoint checkpoints/step_500.pt "Hello"
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mini-SLM: train and generate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the training loop")
    train_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to training config (default: config.yaml)",
    )

    gen_parser = subparsers.add_parser("generate", help="Generate text from a checkpoint")
    gen_parser.add_argument("prompt", nargs="?", default="Once upon a time", help="Prompt text")
    gen_parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    gen_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    gen_parser.add_argument("--max-new-tokens", type=int, default=None)
    gen_parser.add_argument("--temperature", type=float, default=None)
    gen_parser.add_argument("--top-k", type=int, default=None)
    gen_parser.add_argument("--top-p", type=float, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from train import train

        train(args.config)
        return

    if args.command == "generate":
        from generate import generate

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
        return

    parser.print_help()


if __name__ == "__main__":
    main()
