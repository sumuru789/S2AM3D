import argparse
import os.path as osp
from datetime import datetime
import pytz

def default_argument_parser(add_help=True, default_config_file=""):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config-file", '-c', default=default_config_file, metavar="FILE", help="path to config file")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def setup(args, freeze=True):
    from .defaults import _C as cfg
    cfg = cfg.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    dt = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%y%m%d-%H%M%S')
    cfg.output_dir = osp.join(cfg.output_dir, cfg.name, dt)
    if freeze:
        cfg.freeze()
    return cfg