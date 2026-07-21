def make_seg_head(config):
    if config.seg_head.name == "mlp":
        from . import seghead_mlp

        return seghead_mlp.make(config)
    raise NotImplementedError("seg_head %s not supported." % config.seg_head.name)


def make_PointFeatureEnhancer(config):
    from . import PointFeatureEnhancer

    return PointFeatureEnhancer.make(config)


def make_decoder(config):
    from . import CrossAttentionDecoder

    return CrossAttentionDecoder.make(config)
