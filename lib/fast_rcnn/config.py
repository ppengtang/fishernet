# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Peng Tang for Deep Patch Learning
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's longest side
__C.TRAIN.SCALES = (336, 504, 672, 840, 1008)

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 320000

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 1000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

__C.TRAIN.SIGMOID_CROSS_ENTROPY_LOSS = False

#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's longest side
__C.TEST.SCALES = (336, 504, 672, 840, 1008)

# Use horizontally-flipped images during testing?
__C.TEST.USE_FLIPPED = False

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

__C.WHOLE_IMAGE_FINETUNE = False

def get_output_dir(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
