from .tan import TAN
ARCHITECTURES = {"TAN": TAN}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
