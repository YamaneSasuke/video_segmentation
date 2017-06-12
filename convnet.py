# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 16:30:07 2017

@author: yamane
"""

import os
import numpy as np
import time
import tqdm
import copy
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L

import load_datasets


# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            conv1_1=L.ConvolutionND(3, 3, 64, 3, pad=1, stride=1),
            conv2_1=L.ConvolutionND(3, 64, 128, 3, pad=1, stride=1),
            conv3_1=L.ConvolutionND(3, 128, 256, 3, pad=1, stride=1),
            conv3_2=L.ConvolutionND(3, 256, 256, 3, pad=1, stride=1),
            conv4_1=L.ConvolutionND(3, 256, 256, 3, pad=1, stride=1),
            conv4_2=L.ConvolutionND(3, 256, 256, 3, pad=1, stride=1),
            conv5_1=L.ConvolutionND(3, 256, 256, 3, pad=1, stride=1),
            conv5_2=L.ConvolutionND(3, 256, 256, 3, pad=1, stride=1),

            l6=L.Linear(4096, 4096),
            l7=L.Linear(4096, 101)
        )

    def network(self, x):
        h = self.conv1_1(x)
        h = F.max_pooling_nd(h, ksize=(1, 2, 2), stride=(1, 2, 2))
        h = self.conv2_1(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.l6(h)
        y = self.l7(h)
        return y

    def forward(self, x):
        y = self.network(x)
        return y

    def lossfun(self, x, t, test):
        y = self.forward(x)
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def loss_ave(self, queue, num_batches, test):
        losses = []
        for i in range(num_batches):
            X_batch, T_batch = queue.get()
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
        return np.mean(losses)

    def predict(self, x, test):
        y = self.forward(x)
        return F.softmax(y)


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    time_start = time.time()
    epoch_loss = []
    epoch_valid_loss = []
    loss_valid_best = np.inf
    t_loss = []

    # 超パラメータ
    max_iteration = 1500000  # 繰り返し回数
    batch_size = 30  # ミニバッチサイズ
    num_train = 10000  # 学習データ数
    num_valid = 1000  # 検証データ数
    learning_rate = 0.003  # 学習率
    crop_size = 112  # ネットワーク入力画像サイズ
    trim_size = 16
    # 学習結果保存場所
    output_location = r'C:\Users\yamane\Dropbox\M1\video_segmentation'
    # 学習結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, file_name)
    folder_name = str(time_start)
    output_root_dir = os.path.join(output_root_dir, folder_name)
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)
    # ファイル名を作成
    model_filename = str(file_name) + '.npz'
    loss_filename = 'epoch_loss' + str(time_start) + '.png'
    t_dis_filename = 't_distance' + str(time_start) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)
    t_dis_filename = os.path.join(output_root_dir, t_dis_filename)
    # バッチサイズ計算
    num_batches_train = int(num_train / batch_size)
    num_batches_valid = int(num_valid / batch_size)
    # stream作成
    streams = load_datasets.load_ucf101_stream(
        batch_size, num_train, num_batches_valid)
    train_stream, valid_stream, test_stream = streams
    # キューを作成、プロセススタート
    queue_train = Queue(10)
    process_train = Process(target=load_datasets.load_data,
                            args=(queue_train, train_stream, crop_size,
                                  trim_size))
    process_train.start()
    queue_valid = Queue(10)
    process_valid = Process(target=load_datasets.load_data,
                            args=(queue_valid, valid_stream, crop_size,
                                  trim_size))
    process_valid.start()
    # モデル読み込み
    model = Convnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            for i in tqdm.tqdm(range(num_batches_train)):
                X_batch, T_batch = queue_train.get()
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss = model.lossfun(X_batch, T_batch, False)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))

            loss_valid = model.loss_ave(queue_valid, num_batches_valid, True)
            epoch_valid_loss.append(loss_valid)
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                epoch__loss_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print ()
            print ("epoch:", epoch)
            print ("time", epoch_time, "(", total_time, ")")
            print ("loss[train]:", epoch_loss[epoch])
            print ("loss[valid]:", loss_valid)
            print ("loss[valid_best]:", loss_valid_best)
            print ("epoch[valid_best]:", epoch__loss_best)

            if (epoch % 10) == 0:
                plt.figure(figsize=(16, 12))
                plt.plot(epoch_loss)
                plt.plot(epoch_valid_loss)
                plt.ylim(0, 0.5)
                plt.title("loss")
                plt.legend(["train", "valid"], loc="upper right")
                plt.grid()
                plt.show()

    except KeyboardInterrupt:
        print ("割り込み停止が実行されました")

    plt.figure(figsize=(16, 12))
    plt.plot(epoch_loss)
    plt.plot(epoch_valid_loss)
    plt.ylim(0, 0.5)
    plt.title("loss")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    process_train.terminate()
    process_valid.terminate()
    print ('max_iteration:', max_iteration)
    print ('learning_rate:', learning_rate)
    print ('batch_size:', batch_size)
    print ('train_size', num_train)
    print ('valid_size', num_valid)
    print ('crop_size', crop_size)
    print ('trim_size', trim_size)
