# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:20:23 2017

@author: yamane
"""


import os
import numpy as np
import h5py
from Queue import Full

import fuel
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme

import utility


def load_ucf101_stream(batch_size, train_size=16500, validation_size=500,
                       test_size=100, shuffle=False):
    fuel_root = fuel.config.data_path[0]
    # データセットファイル保存場所
    hdf5_filepath = os.path.join(
        fuel_root, 'UCF101\hdf5_dataset\hdf5_dataset.hdf5')
    valid_size = train_size + validation_size
    test_size = valid_size + test_size
    indices_train = range(0, train_size)
    indices_valid = range(train_size, valid_size)
    indices_test = range(valid_size, test_size)

    h5py_file = h5py.File(hdf5_filepath)
    dataset = H5PYDataset(h5py_file, ['train'])

    scheme_class = ShuffledScheme if shuffle else SequentialScheme
    scheme_train = scheme_class(indices_train, batch_size=batch_size)
    scheme_valid = scheme_class(indices_valid, batch_size=batch_size)
    scheme_test = scheme_class(indices_test, batch_size=batch_size)

    stream_train = DataStream(dataset, iteration_scheme=scheme_train)
    stream_valid = DataStream(dataset, iteration_scheme=scheme_valid)
    stream_test = DataStream(dataset, iteration_scheme=scheme_test)
    stream_train.get_epoch_iterator().next()
    stream_valid.get_epoch_iterator().next()
    stream_test.get_epoch_iterator().next()

    return stream_train, stream_valid, stream_test


def data_crop(X_batch, T_batch, crop_size=112, trim_size=16):
    videos = []
    targets = []

    for b in range(X_batch.shape[0]):
        # 補間方法を乱数で設定
        video = X_batch[b]
        trim_video = utility.random_trim(video, trim_size)
        crop_video = utility.random_crop_and_flip(trim_video, crop_size)
        videos.append(np.transpose(crop_video, (1, 0, 2, 3)))
        targets.append(T_batch[b])
    batch = np.stack(videos, axis=0)
    batch = batch.astype(np.float32)
    target = np.array(targets, dtype=np.float32).reshape(-1, 1)
    return batch, target


def load_data(queue, stream, crop_size=112, trim_size=16, random=True):
    while True:
        for T, X in stream.get_epoch_iterator():
            X, T = data_crop(X, T, crop_size, trim_size)
            queue.put((X, T))


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 150000  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 10000  # 学習データ数
    num_valid = 1000  # 検証データ数
    learning_rate = 0.003  # 学習率
    crop_size = 112  # ネットワーク入力画像サイズ
    trim_size = 16
    # バッチサイズ計算
    num_batches_train = int(num_train / batch_size)
    num_batches_valid = int(num_valid / batch_size)
    # stream作成
    streams = load_ucf101_stream(batch_size, num_train, num_batches_valid)
    train_stream, valid_stream, test_stream = streams

    for T, X in train_stream.get_epoch_iterator():
        X, T = data_crop(X, T, crop_size, trim_size)
        print X.shape
        print T.shape
