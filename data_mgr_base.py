import os
import random
from threading import Thread

import numpy as np


class DataWrapper(object):

    def __init__(self):
        pass

    def access(self, start_idx, end_idx):
        pass

    def size(self):
        pass

    def shuffle(self):
        pass


class ListDataWrapper(DataWrapper):

    def __init__(self, list_data):
        self._list_data = list_data

    def access(self, start_idx, end_idx):
        return self._list_data[start_idx:end_idx]

    def size(self):
        return len(self._list_data)

    def shuffle(self):
        random.shuffle(self._list_data)


class DataBatcher(object):

    def __init__(self, data_wrapper):
        self._data_wrapper = data_wrapper
        self._epoch = 0
        self._curr_idx = 0

    def _next_batch_impl(self, batch_size):
        end_idx = self._curr_idx + batch_size
        if end_idx > self._data_wrapper.size():
            self._data_wrapper.shuffle()
            self._epoch += 1
            self._curr_idx = 0
            end_idx = batch_size
        ans = self._data_wrapper.access(self._curr_idx, end_idx)
        self._curr_idx = end_idx
        return ans

    def next_batch(self, batch_size):
        ans = []
        while batch_size > 0:
            ans.extend(self._next_batch_impl(batch_size))
            batch_size -= len(ans)
        return ans
    
    def epoch(self):
        return self._epoch

    def size(self):
        return self._data_wrapper.size()

    def progress(self):
        return self._curr_idx * 1.0 / self._data_wrapper.size()


def stack_list(list_data):
    return np.concatenate(
        [np.expand_dims(x, axis=0) for x in list_data], axis=0)


def make_list_data_batcher(list_data):
    return DataBatcher(ListDataWrapper(list_data))


class DataMgr(object):

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def _train_batch(self):
        pass

    def _valid_batch(self):
        pass

    def _test_batch(self):
        pass

    def train_batch(self):
        return self._train_batch()

    def valid_batch(self):
        return self._valid_batch()

    def test_batch(self):
        return self._test_batch()


class _DataLoadingWorker(Thread):

    def __init__(self, data_loading_fn, num_loads):
        Thread.__init__(self)
        self._data_loading_fn = data_loading_fn
        self._num_loads = num_loads
        self._data = []
        
    def run(self):
        for i in range(self._num_loads):
            self._data.append(self._data_loading_fn())


class _DataLoadingMgr(object):

    def __init__(self, data_loading_fn, cache_size):
        self._data_loading_fn = data_loading_fn
        self._cache_size = cache_size
        if not self._small_cache_size():
            self._worker = None
            self._data = [
                data_loading_fn() for i in range(cache_size)
            ]
            self._start_worker()

    def _small_cache_size(self):
        return self._cache_size <= 1

    def _start_worker(self):
        if self._worker is None:
            self._worker = _DataLoadingWorker(
                self._data_loading_fn, self._cache_size)
            self._worker.start()

    def _join_worker(self):
        if self._worker is not None:
            self._worker.join()
            self._data = self._worker._data
            self._worker = None
    
    def get(self):
        if self._small_cache_size():
            return self._data_loading_fn()
        if self._data:
            return self._data.pop()
        self._join_worker()
        ans = self._data.pop()
        self._start_worker()
        return ans
        
        
class CacheDataMgr(DataMgr):

    def __init__(self, batch_size, cache_sizes):
        super(CacheDataMgr, self).__init__(batch_size)

        self._train_cache_size = cache_sizes[0]
        self._valid_cache_size = cache_sizes[1]
        self._test_cache_size = cache_sizes[2]

    def train_batch(self):
        if not hasattr(self, "_train_mgr"):
            self._train_mgr = _DataLoadingMgr(
                lambda: self._train_batch(), self._train_cache_size)
        return self._train_mgr.get()

    def valid_batch(self):
        if not hasattr(self, "_valid_mgr"):
            self._valid_mgr = _DataLoadingMgr(
                lambda: self._valid_batch(), self._valid_cache_size)
        return self._valid_mgr.get()

    def test_batch(self):
        if not hasattr(self, "_test_mgr"):
            self._test_mgr = _DataLoadingMgr(
                lambda: self._test_batch(), self._test_cache_size)
        return self._test_mgr.get()


class FileProp(object):

    def __init__(self):
        pass

    def is_valid_file_name(self, file_name):
        return True

    def read_file(self, file_path):
        return None

    
class FileDataMgr(CacheDataMgr):

    def __init__(self, batch_size, cache_sizes,
                 file_dirs, file_prop, train_valid_test_splits):
        super(FileDataMgr, self).__init__(batch_size, cache_sizes)

        all_file_paths = []
        for file_dir in file_dirs:
            if not os.path.isdir(file_dir):
                continue

            all_file_paths.extend([
                (file_dir, file_name) for file_name in os.listdir(file_dir)
                if file_prop.is_valid_file_name(file_name)
            ])

        num_files = len(all_file_paths)
        train_acc_pct, valid_acc_pct, test_acc_pct = np.cumsum(
            train_valid_test_splits)
        train_acc_size = int(num_files * train_acc_pct)
        valid_acc_size = int(num_files * valid_acc_pct)
        test_acc_size = int(num_files * test_acc_pct)
        ok = (train_acc_size <= 0 or
              valid_acc_size <= train_acc_size or
              test_acc_size <= valid_acc_size)
        assert not ok, ("Invalid train / valid / test: {}".format(
            train_valid_test_splits))
        
        self._file_prop = file_prop
        self._train_valid_test_splits = train_valid_test_splits
        
        random.shuffle(all_file_paths)
        self._all_file_paths = all_file_paths
        self._train_batcher = make_list_data_batcher(
            all_file_paths[0:train_acc_size])
        self._valid_batcher = make_list_data_batcher(
            all_file_paths[train_acc_size:valid_acc_size])
        self._test_batcher = make_list_data_batcher(
            all_file_paths[valid_acc_size:test_acc_size])

    def _get_batch(self, data_batcher):
        list_data = []
        count = 0
        while count < 100 and len(list_data) < self._batch_size:
            n = self._batch_size - len(list_data)
            file_paths = data_batcher.next_batch(n)
            for file_dir, file_name in file_paths:
                try:
                    list_data.append(self._file_prop.read_file(
                        os.path.join(file_dir, file_name)))
                except Exception as e:
                    print "Error: {}".format(e)
            count += 1
        list_data = list_data[0:self._batch_size]
        assert len(list_data) == self._batch_size
        random.shuffle(list_data)
        return list_data

    def _train_batch(self):
        return self._get_batch(self._train_batcher)

    def _valid_batch(self):
        return self._get_batch(self._valid_batcher)

    def _test_batch(self):
        return self._get_batch(self._test_batcher)
