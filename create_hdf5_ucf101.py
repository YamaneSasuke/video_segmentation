# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:18:12 2017

@author: yamane
"""

import os
import sys
import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
import tqdm
import cv2


def create_path_list(dataset_root_dir):
    path_list = []

    for root, dirs, files in os.walk(dataset_root_dir):
        for file_name in tqdm.tqdm(files):
            file_path = os.path.join(root, file_name)
            path_list.append(file_path)
    return path_list


def output_path_list(path_list, output_root_dir):
    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.txt'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = open(output_root_dir, "w")
    for path in path_list:
        f.write(path + "\n")
    f.close()


def output_hdf5(path_list, output_root_dir):
    num_data = len(path_list)
#    shapes = []
    class_list = []

    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.hdf5'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = h5py.File(output_root_dir, mode='w')
    dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
    video_features = f.create_dataset('video_features',
                                      (num_data,),
                                      dtype=dtype)
    targets = f.create_dataset('targets', (num_data,), dtype='uint8')
    video_features_shapes = f.create_dataset('video_features_shapes',
                                             (num_data, 4), dtype=np.int32)

    video_features.dims[0].label = 'batch'
    targets.dims[0].label = 'batch'

    for path in path_list:
        path = path.strip()
        dirs = path.split('\\')
        ucf101_index = dirs.index('UCF-101')
        class_list.append('_'.join(dirs[ucf101_index+1:-1]))
    class_uniq = list(set(class_list))

    try:
        for i in tqdm.tqdm(range(num_data)):
            cap = cv2.VideoCapture(path_list[i])
            frames = []
            for frame_n in range(1000):
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
            video = np.stack(frames, axis=0)
#            shapes.append(video.shape)
            video_features[i] = video.flatten()
            targets[i] = class_uniq.index(class_list[i])
            video_features_shapes[i] = video.shape
            cap.release()
            cv2.destroyAllWindows()

#        shapes = np.array(shapes).astype(np.int32)
#        video_features_shapes[...] = shapes

        video_features.dims.create_scale(video_features_shapes, 'shapes')
        video_features.dims[0].attach_scale(video_features_shapes)

        video_features_shape_labels = f.create_dataset(
            'video_features_shape_labels', (4,), dtype='S7')
        video_features_shape_labels[...] = [
             'frame'.encode('utf8'), 'channel'.encode('utf8'),
             'height'.encode('utf8'), 'width'.encode('utf8')]
        video_features.dims.create_scale(
            video_features_shape_labels, 'shape_labels')
        video_features.dims[0].attach_scale(video_features_shape_labels)

        # specify the splits
        split_train = (0, num_data)
        split_dict = dict(train=dict(video_features=split_train,
                                     targets=split_train))
        f.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    except KeyboardInterrupt:
        print ("割り込み停止が実行されました")

    f.flush()
    f.close()


def main(data_location):
    # hdf５ファイルを保存する場所
    output_root_dir = os.path.join(data_location, 'hdf5_dataset')
    dataset_root_dir = os.path.join(data_location, 'UCF-101')

    if os.path.exists(output_root_dir):
        print (u"すでに存在するため終了します.")
        sys.exit()
    else:
        os.makedirs(output_root_dir)

    path_list = create_path_list(dataset_root_dir)
    shuffled_path_list = np.random.permutation(path_list)
    output_path_list(shuffled_path_list, output_root_dir)
    output_hdf5(shuffled_path_list, output_root_dir)


if __name__ == '__main__':
    # PASCALVOC2012データセットの保存場所
    data_location = r'E:\UCF101'

    main(data_location)
