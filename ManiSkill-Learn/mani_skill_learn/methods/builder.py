from ..utils.meta import Registry, build_from_cfg

MBRL = Registry('mbrl')
MFRL = Registry('mfrl')  # Model free RL
BRL = Registry('brl')  # Offline RL / Batch RL



def build_mfrl(cfg, default_args=None):
    return build_from_cfg(cfg, MFRL, default_args)


def build_brl(cfg, default_args=None):
    return build_from_cfg(cfg, BRL, default_args)


def build_mbrl(cfg,default_args=None):
    return build_from_cfg(cfg, MBRL, default_args)
