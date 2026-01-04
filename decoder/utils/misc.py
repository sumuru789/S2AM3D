from omegaconf import OmegaConf


def load_config(*yaml_files, cli_args=None, extra_args=None):
    cli_args = cli_args or {}
    extra_args = extra_args or []
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    yaml_confs.append(OmegaConf.from_cli(extra_args))
    conf = OmegaConf.merge(*yaml_confs, cli_args)
    OmegaConf.resolve(conf)
    return conf
