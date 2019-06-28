import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import skimage
import json


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def heatmaps_to_coords(heatmaps, resolu_out=[64,64], prob_threshold=0.2):
    '''
    :param heatmaps: tensor with shape (64,64,16)
    :param resolu_out: output resolution list
    :return coord_joints: np array, shape (16,2)
    '''

    num_joints = heatmaps.shape[2]

    if tf.is_tensor(heatmaps) == True:
        heatmaps = heatmaps.numpy()

    # Resize
    heatmaps = skimage.transform.resize(heatmaps, tuple(resolu_out))
    print('heatmaps.SHAPE', heatmaps.shape)

    coord_joints = np.zeros((num_joints, 2))
    for i in range(num_joints):
        heatmap = heatmaps[..., i]
        max = np.max(heatmap)
        # Only keep points larger than a threshold
        if max > prob_threshold:
            idx = np.where(heatmap == max)
            H = idx[0][0]
            W = idx[1][0]
        else:
            H = 0
            W = 0
        coord_joints[i] = [W, H]
        # print('x', x)
        # print('y', y)
    # print('coord_joints', coord_joints)

    return coord_joints

#==================== Test ====================
if __name__ == '__main__':
    a = np.zeros((64, 64, 16))
    a[9, 6, 0] = 7
    a[7, 13, 0] = 3
    a[8, 8, 1] = 6
    heatmaps_to_coords(a)