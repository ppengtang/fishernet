import sys
import os
import argparse
import scipy.io as sio
import numpy as np

import _init_paths
import caffe

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--train', dest='train',
                        help='train prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--parameters', dest='parameters_file',
                        help='initialize with parameters',
                        default=None, type=str)
    parser.add_argument('--output', dest='filename',
                        help='output dir',
                        default='initialized.caffemodel', type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('begin')
    args = parse_args()

    print('Called with args:')
    print(args)

    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    net = caffe.Net(args.train, args.pretrained_model, caffe.TRAIN)

    while not os.path.exists(args.parameters_file) and args.wait:
        print('Waiting for {} to exist...'.format(args.parameters_file))
        time.sleep(10)

    parameters = sio.loadmat(args.parameters_file)
    weights = parameters['weights'].reshape(-1, 1)
    bias = parameters['bias'].reshape(-1, 1)
    priors = parameters['priors_new'].reshape(-1, 1)
    pca_w = parameters['pca_w'].T.reshape(-1, 1)
    pca_b = parameters['pca_b'].T.reshape(-1, 1)

    assert len(weights) == len(net.params['fisher1'][0].data) \
      and len(bias) == len(net.params['fisher1'][1].data) \
      and len(weights) == len(bias) \
      and len(net.params['fisher1'][0].data) == len(net.params['fisher1'][1].data) \
      and len(priors) == len(net.params['fisher_weight'][0].data), \
      'dimension must be equal!'

    weights = weights.reshape(net.params['fisher1'][0].data.shape)
    bias = bias.reshape(net.params['fisher1'][1].data.shape)
    pca_w = pca_w.reshape(net.params['res5c_pca'][0].data.shape)
    pca_b = pca_b.reshape(net.params['res5c_pca'][1].data.shape)

    for i in xrange(len(weights)):
    	net.params['fisher1'][0].data[i] = weights[i]
    	net.params['fisher1'][1].data[i] = bias[i]
    for i in xrange(len(priors)):
        net.params['fisher_weight'][0].data[i] = priors[i]

    net.params['res5c_pca'][0].data[...] = pca_w[...]
    net.params['res5c_pca'][1].data[...] = pca_b[...]

    net.save(str(args.filename))
