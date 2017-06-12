# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:45:30 2017

@author: yamane
"""

import numpy as np


def random_crop_and_flip(video, crop_size=112):
    h_frame, w_frame = video.shape[2:4]
    h_crop = crop_size
    w_crop = crop_size

    # 0以上 h_image - h_crop以下の整数乱数
    top = np.random.randint(0, h_frame - h_crop + 1)
    left = np.random.randint(0, w_frame - w_crop + 1)
    bottom = top + h_crop
    right = left + w_crop

    new_video = []
    for i in range(video.shape[0]):
        new_video.append(video[i, :, top:bottom, left:right])
    new_video = np.stack(new_video, axis=0)

    if np.random.rand() > 0.5:  # 半々の確率で
        new_video = new_video[:, :, :, ::-1]  # 左右反転

    return new_video


def random_trim(video, trim_size=16):
    num_frame = video.shape[0]

    # 0以上 h_image - h_crop以下の整数乱数
    start = np.random.randint(0, num_frame - trim_size)

    new_video = video[start:start+trim_size]

    return new_video
