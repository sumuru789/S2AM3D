import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the decoder config file.",
    )
    args, extras = parser.parse_known_args(args)
    return args, extras
