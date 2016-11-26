import os
import random
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize

import data_mgr_base


_MY_DIR = os.path.dirname(os.path.realpath(__file__))


def _resolve_file_dirs(file_dirs):
    if file_dirs is None:
        file_dirs = "../aflw/aflw/data/flickr_face"
    if isinstance(file_dirs, basestring):
        return [file_dirs]
    return file_dirs


def _process_image(im, new_w, new_h, force_grayscale):
    h, w = im.shape[:2]
    a = min(h, w)
    y, x = h // 2 - a // 2, w // 2 - a // 2
    im = im[y:y+a, x:x+a, 0:3]
    im = imresize(im, [new_w, new_h]) / 255.0
    if force_grayscale and im.shape[2] == 3:
        return np.dot(im, [0.299, 0.587, 0.114])
    assert im.shape[2] in (1, 3)
    return im


class _FileProp(data_mgr_base.FileProp):

    def __init__(self, image_width, image_height, force_grayscale):
        super(_FileProp, self).__init__()
        self._image_width = image_width
        self._image_height = image_height
        self._force_grayscale = force_grayscale

    def is_valid_file_name(self, file_name):
        return file_name.endswith(".png")

    def read_file(self, file_path):
        im = imread(file_path)
        return _process_image(im, self._image_width, self._image_height,
                              self._force_grayscale)


class DataMgr(data_mgr_base.FileDataMgr):

    def __init__(self, batch_size,
                 cache_sizes=[32, 2, 2],
                 image_width=64, image_height=64, force_grayscale=False,
                 file_dirs=None, train_valid_test_splits=[0.8, 0.1, 0.1]):
        file_dirs = _resolve_file_dirs(file_dirs)
        super(DataMgr, self).__init__(
            batch_size=batch_size, cache_sizes=cache_sizes,
            file_dirs=file_dirs,
            file_prop=_FileProp(image_width, image_height, force_grayscale),
            train_valid_test_splits=train_valid_test_splits)
                
        
    def _get_batch(self, data_batcher):
        list_data = super(DataMgr, self)._get_batch(data_batcher)
        return data_mgr_base.stack_list(list_data)
