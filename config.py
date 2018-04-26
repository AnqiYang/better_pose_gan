from easydict import EasyDict as ed
import os

cfg = ed()

cfg.IMAGE_SHAPE = [256, 256, 3]  #todo: change back
cfg.G1_INPUT_DATA_SHAPE = cfg.IMAGE_SHAPE[:2] + [21]
cfg.BATCH_SIZE = 8
cfg.BATCH_SIZE_G2D = 8
cfg.N = 6  # number of residual blocks
cfg.WEIGHT_DECAY = 1e-4
cfg.LAMBDA = 10
cfg.MAXITERATION1 = 22000
cfg.MAXITERATION2 = 11000
cfg.LOGDIR = './logs'
cfg.MODE = 'train'
cfg.RESULT_DIR = './result'
