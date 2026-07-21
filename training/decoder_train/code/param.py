import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default="decoder_train",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./exp",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--train",
        default=True,
        action="store_true",
        help="train a model."
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume from (default: train from scratch)."
    )
    parser.add_argument(
        "--autoresume",
        default=False,
        action="store_true",
        help="auto back-off on failure"
    )
    parser.add_argument(
        "--ngpu", 
        default=2,
        type=int,
        help="number of gpu used"
    )
    args, extras = parser.parse_known_args()
    return args, extras