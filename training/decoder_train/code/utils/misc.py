from omegaconf import OmegaConf


def load_config(*yaml_files, cli_args=[], extra_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    yaml_confs += [OmegaConf.from_cli(extra_args)]
    conf = OmegaConf.merge(*yaml_confs, cli_args)
    OmegaConf.resolve(conf)
    return conf


def dump_config(path, config):
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)
