def make_seg_head(config):

    if config.seg_head.name == "mlp":
        from . import seghead_mlp
        model = seghead_mlp.make(config)
    else:
        raise NotImplementedError("seg_head %s not supported." % config.seg_head.name)
    return model


def make_PointFeatureEnhancer(config):

    from . import PointFeatureEnhancer
    model = PointFeatureEnhancer.make(config)
    return model


def make_decoder(config):

    from . import CrossAttentionDecoder
    model = CrossAttentionDecoder.make(config)
    return model


