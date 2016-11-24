import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import data_mgr_base


class DataMgr(data_mgr_base.DataMgr):

    def __init__(self, batch_size, binarize=False, data_dir="mnist_data"):
        super(DataMgr, self).__init__(batch_size)
        self._binarize = binarize
        self._data = input_data.read_data_sets(data_dir, one_hot=False)

    def _get_batch(self, data):
        ans = data.next_batch(self._batch_size)[0]
        ans = np.reshape(ans, [-1, 28, 28])
        if self._binarize:
            ans = (ans >= 0.5).astype(np.float32)
        return ans

    def _train_batch(self):
        return self._get_batch(self._data.train)

    def _valid_batch(self):
        return self._get_batch(self._data.validation)

    def _test_batch(self):
        return self._get_batch(self._data.test)
