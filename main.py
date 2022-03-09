import os
import sys

import pytorch_lightning as pl
import torch

from faceformer.config import get_cfg, get_parser
from faceformer.datasets import *
from faceformer.models import *
from faceformer.trainer import Trainer


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_parser().parse_args()

    cfg = get_cfg(args)
    
    model_class = str_to_class(cfg.model_class)
    dataset_class = str_to_class(cfg.dataset_class)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        filename='{epoch:d}-{valid_precision:.2f}',
        save_top_k=2,
        monitor='valid_precision',
        mode='max',
        every_n_val_epochs=1)

    logger = pl.loggers.TensorBoardLogger('logs/', name=cfg.trainer.name, version=cfg.trainer.version)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(c) for c in cfg.trainer.num_gpus])
    gpus = list(range(len(cfg.trainer.num_gpus)))

    if args.test_ckpt != '':
        # Testing
        model = Trainer(cfg, model_class, dataset_class).load_from_checkpoint(args.test_ckpt, model_class=model_class, dataset_class=dataset_class)
        trainer = pl.Trainer(
            benchmark=True,
            gpus=gpus,
            precision=cfg.trainer.precision)
        trainer.test(model)
    elif args.valid_ckpt != '':
        # Validation
        model = Trainer(cfg, model_class, dataset_class).load_from_checkpoint(args.valid_ckpt, model_class=model_class, dataset_class=dataset_class)
        trainer = pl.Trainer(
            benchmark=True,
            gpus=gpus,
            precision=cfg.trainer.precision)
        trainer.validate(model)
    elif args.resume_ckpt != '':
        # Resume Training
        model = Trainer(cfg, model_class, dataset_class).load_from_checkpoint(args.resume_ckpt, model_class=model_class, dataset_class=dataset_class)
        trainer = pl.Trainer(
            logger=logger,
            benchmark=True,
            gpus=gpus,
            precision=cfg.trainer.precision,
            resume_from_checkpoint=args.resume_ckpt)
        trainer.fit(model)
    else:
        model = Trainer(cfg, model_class, dataset_class)
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[checkpoint_callback, CudaClearCacheCallback()],
            check_val_every_n_epoch=cfg.trainer.checkpoint_period,
            log_every_n_steps=1,
            benchmark=True,
            gpus=gpus,
            precision=cfg.trainer.precision)
        trainer.fit(model)
