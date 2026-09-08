"""Command-line interface for training and evaluation."""

import argparse

from .config import DEFAULT_CONFIG_PATH

COMMANDS = {
    "train-centralized": "Train the centralized baseline.",
    "train-federated": "Train the FedAvg federated model.",
    "evaluate": "Evaluate saved checkpoints and regenerate comparison figures.",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the ``xray-fl`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="xray-fl",
        description="Chest X-ray pneumonia classification: centralized vs. federated learning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, help_text in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        subparser.add_argument(
            "--config",
            default=DEFAULT_CONFIG_PATH,
            help=f"Path to the YAML config (default: {DEFAULT_CONFIG_PATH}).",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand to its entry point."""
    args = build_parser().parse_args(argv)

    # Imported lazily so that ``xray-fl --help`` stays fast and torch-free.
    if args.command == "train-centralized":
        from .train_centralized import main as run
    elif args.command == "train-federated":
        from .train_federated import main as run
    else:
        from .evaluate import main as run

    run(config_path=args.config)
    return 0
