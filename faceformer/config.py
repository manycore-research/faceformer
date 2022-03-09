import argparse

from fvcore.common.config import CfgNode

CN = CfgNode

_C = CN()
_C.model_class = 'SurfaceFormer'
_C.dataset_class = 'ABCDataset'
_C.root_dir = "/root/data"

_C.batch_size_train = 64
_C.batch_size_valid = 128
_C.datasets_train = ['train.txt']
_C.datasets_valid = ['valid.txt']
_C.datasets_test = ['test.txt']

_C.trainer = CN()
_C.trainer.name = "surfaceformer"
_C.trainer.version = "baseline"
_C.trainer.num_gpus = [0]
_C.trainer.precision = 16 # 16-bit training
_C.trainer.checkpoint_period = 2
_C.trainer.lr = 1e-3
_C.trainer.lr_step = 0

_C.model = CN()
_C.model.num_points_per_line = 50
_C.model.num_lines = 64
_C.model.point_dim = 2
_C.model.label_seq_length = 128
_C.model.max_num_faces = 42
_C.model.max_face_length = 34
_C.model.num_model = 512
_C.model.num_head = 8
_C.model.num_feedforward = 1024
_C.model.num_encoder_layers = 6
_C.model.num_decoder_layers = 6
_C.model.dropout = 0.2
_C.model.token = CN()
_C.model.token.PAD = 0
_C.model.token.SOS = 1
_C.model.token.SEP = 2
_C.model.token.EOS = 3
_C.model.token.DIR0 = 4
_C.model.token.DIR1 = 5
_C.model.token.len = 4
_C.model.token.face_type_offset = 1

_C.post_process = CN()
_C.post_process.enclosedness_tol = 2e-4
_C.post_process.is_coedge = True

def get_parser():
    parser = argparse.ArgumentParser(description="SurfaceFormer Training")
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--valid_ckpt", default="",
                        help="path to validation checkpoint")
    parser.add_argument("--test_ckpt", default="",
                        help="path to testing checkpoint")
    parser.add_argument("--resume_ckpt", default="",
                        help="path to training checkpoint, will continue train from here")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def get_cfg(args):
    cfg = _C.clone()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
