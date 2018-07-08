# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Peng Tang for Deep Patch Learning
# --------------------------------------------------------

"""Test a DPL network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import scipy.io as sio

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        im_shapes: the list of image shapes
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    processed_ims = []
    if not cfg.WHOLE_IMAGE_FINETUNE:
        im_size_max = np.max(im_shape[0:2])
        # im_size_min = np.min(im_shape[0:2])

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_max)
            # if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            #     im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            processed_ims.append(im_list_to_blob([im]))
    else:
        im_scale_x = 224.0 / im_shape[1]
        im_scale_y = 224.0 / im_shape[0]
        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im_list_to_blob([im]))

    blob = processed_ims

    return blob

def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data']= _get_image_blob(im)
    return blobs

def patch_feature(net, im):
    """Get feature of the given image.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        feature (ndarray): 1 x K array of image features
    """
    blobs = _get_blobs(im)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    for i in xrange(len(blobs['data'])):
        net.blobs['data'].reshape(*(blobs['data'][i].shape))
        blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False))
        feature_tmp = blobs_out['res5c']
        feature_tmp = feature_tmp.T
        feature_tmp = feature_tmp.reshape(feature_tmp.shape[0]*feature_tmp.shape[1], feature_tmp.shape[2])

        if i == 0:
            feature = np.copy(feature_tmp)
        else:
            feature = np.vstack((feature, feature_tmp))

    return feature

def im_feature(net, im):
    """Get feature of the given image.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        feature (ndarray): 1 x K array of image features
    """
    blobs = _get_blobs(im)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    for i in xrange(len(blobs['data'])):
        net.blobs['data'].reshape(*(blobs['data'][i].shape))
        blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False))
        feature_tmp = blobs_out['fisher_sum'].reshape(1, -1)
        #feature_tmp = feature_tmp.T
        #feature_tmp = feature_tmp.reshape(feature_tmp.shape[0]*feature_tmp.shape[1], feature_tmp.shape[2])

        if i == 0:
            feature = np.copy(feature_tmp)
        else:
            # feature = np.vstack((feature, feature_tmp))
            feature += feature_tmp

        if cfg.TEST.USE_FLIPPED:
            blobs['data'][i] = blobs['data'][i][:, :, :, ::-1]
            blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False))
            feature += blobs_out['fisher_sum'].reshape(1, -1)

    # feature = np.max(feature, axis=0)

    return feature

def feature_patch(net, imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'feature_extr' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['feature_extr'].tic()
        feature = patch_feature(net, im)
        _t['feature_extr'].toc()

        _t['misc'].tic()
        feature_path = os.path.join(output_dir, imdb._image_index[i] + '.mat')
        sio.savemat(feature_path, {'feature': feature})
            
        _t['misc'].toc()

        print 'feature_extr: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['feature_extr'].average_time,
                      _t['misc'].average_time)

def feature_image(net, imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'feature_extr' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['feature_extr'].tic()
        feature = im_feature(net, im)
        # feature = np.mean(feature, axis=0)
        _t['feature_extr'].toc()

        _t['misc'].tic()
        feature_path = os.path.join(output_dir, imdb._image_index[i] + '.mat')
        sio.savemat(feature_path, {'feature': feature})
            
        _t['misc'].toc()

        print 'feature_extr: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['feature_extr'].average_time,
                      _t['misc'].average_time)