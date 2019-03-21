import numpy as np
from easydict import EasyDict as edict

config = edict()
#net config
config.img_width = 80
config.img_height = 32
config.num_label = config.img_width//8
config.seq_length = config.img_width//8
#config.image_path = 'images'
config.num_classes = 11
config.num_hidden = 100
config.num_lstm_layer = 2
#config.use_lstm = False
config.use_lstm = True
config.no4x1pooling=False
config.to_gray = True


default = edict()
# default dataset
default.dataset_path = '/home/lironghua/Downloads/data/ocr/Train_data_digit'
# default training
default.lr = 0.001
default.lr_decay = 0.1
default.end_lr = 1e-7
default.lr_decay_step = 15

default.frequent = 20
default.batch_size = 32
default.kvstore = 'device'

default.prefix = 'digit'
default.epoch = 300
default.ctx = '0'
default.save_period = 1
default.save_dir = "./model"
# default.resume_from = 'model/crnn-chinese-lr1-0004.params'
default.resume_from = ''

# network = edict()
# dataset = edict()


