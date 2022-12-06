from yacs.config import CfgNode as CN

def add_MAE_config(cfg):
    cfg.MAE = CN()
    cfg.MAE.PRETRAIN = True
    cfg.MAE.MODEL = 'vit-base'
    cfg.MAE.MASKRATIO = 0.9
    cfg.MAE.ACCUM_ITER = 20
