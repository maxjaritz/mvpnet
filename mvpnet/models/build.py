import warnings

from mvpnet.models.pn2.pn2ssg import PN2SSG
from mvpnet.models.unet_resnet34 import UNetResNet34
from mvpnet.models.mvpnet_3d import MVPNet3D


def build_model_sem_seg_3d(cfg):
    assert cfg.TASK == 'sem_seg_3d', cfg.TASK
    model_fn = globals()[cfg.MODEL.TYPE]
    if cfg.MODEL.TYPE in cfg.MODEL:
        model_cfg = dict(cfg.MODEL[cfg.MODEL.TYPE])
        # loss_cfg = model_cfg.pop('loss', None)
    else:
        warnings.warn('Use default arguments to initialize {}'.format(cfg.MODEL.TYPE))
        model_cfg = dict()
    model = model_fn(**model_cfg)
    loss_fn = model.get_loss(cfg)
    train_metric, val_metric = model.get_metric(cfg)
    return model, loss_fn, train_metric, val_metric


def build_model_sem_seg_2d(cfg):
    assert cfg.TASK == 'sem_seg_2d', cfg.TASK
    model_fn = globals()[cfg.MODEL.TYPE]
    if cfg.MODEL.TYPE in cfg.MODEL:
        model_cfg = dict(cfg.MODEL[cfg.MODEL.TYPE])
        # loss_cfg = model_cfg.pop('loss', None)
    else:
        warnings.warn('Use default arguments to initialize {}'.format(cfg.MODEL.TYPE))
        model_cfg = dict()
    model = model_fn(**model_cfg)
    loss_fn = model.get_loss(cfg)
    train_metric, val_metric = model.get_metric(cfg)
    return model, loss_fn, train_metric, val_metric


def build_model_mvpnet_3d(cfg):
    assert cfg.TASK == 'mvpnet_3d', cfg.TASK
    model_2d_fn = globals()[cfg.MODEL_2D.TYPE]
    model_3d_fn = globals()[cfg.MODEL_3D.TYPE]
    model_2d = model_2d_fn(**cfg.MODEL_2D.get(cfg.MODEL_2D.TYPE, dict()))
    model_3d = model_3d_fn(**cfg.MODEL_3D.get(cfg.MODEL_3D.TYPE, dict()))
    model = MVPNet3D(model_2d, cfg.MODEL_2D.CKPT_PATH, model_3d, **cfg.FEAT_AGGR)
    loss_fn = model.get_loss(cfg)
    train_metric, val_metric = model.get_metric(cfg)
    return model, loss_fn, train_metric, val_metric
