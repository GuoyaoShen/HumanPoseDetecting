import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import imgutils
import os

from utils import imgutils
from utils import imgutilsTF
from utils import Dataset_Make_Utils as dsmutils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Get json annotation
with open('mpii_annotations.json') as anno_file:
    anno = json.load(anno_file)  # len 25204
print(len(anno))

train_list, valid_list = [], []
for idx, ele_anno in enumerate(anno):
    if ele_anno['isValidation'] == True:
        valid_list.append(idx)
    else:
        train_list.append(idx)
print('train LEN:', len(train_list))  #22246
print('valid LEN:', len(valid_list))  #2958

res_heatmap = [256, 256]
res_heatmap = [64, 64]
num_heatmap = 16
# train_imgs = np.zeros((len(train_list), res_img[0], res_img[1], 3))  #(N,H_im,W_im,3)
# train_heatmaps = np.zeros((len(train_list), res_heatmap[0], res_heatmap[1], num_heatmap))  #(N,H_ht,W_ht,16)
# valid_imgs = np.zeros((len(valid_list), res_img[0], res_img[1], 3))  #(N,H_im,W_im,3)
# valid_heatmapss = np.zeros((len(valid_list), res_heatmap[0], res_heatmap[1], num_heatmap))  #(N,H_ht,W_ht,16)
# print('train_imgs.SHAPE', train_imgs.shape)
# print('train_heatmaps.SHAPE', train_heatmaps.shape)
# print('valid_imgs.SHAPE', valid_imgs.shape)
# print('valid_heatmapss.SHAPE', valid_heatmapss.shape)

# Get parent dir path
path_dir = os.getcwd()
path_dir = os.path.dirname(path_dir)
# print(path_dir)

train_heatmaps = np.zeros((1000, res_heatmap[0], res_heatmap[1], 16))  #(N,H_im,W_im,3)
path_img_folder = path_dir+'/mpii_human_pose_v1/images'

num_subset = 5  # from 1 to 22

idx_begin = 1000*(num_subset-1)
idx_end = idx_begin+999
j = 0
for i, idx_train in enumerate(train_list):
    if i >= idx_begin and i <= idx_end:
        j += 1
        print(anno[idx_train])
        ele_anno = anno[idx_train]
        path_img = os.path.join(path_img_folder, ele_anno['img_paths'])
        print(path_img)
        img_origin = imgutils.load_image(path_img)
        # plt.figure(1)
        # plt.imshow(img_origin)
        # plt.show()
        img_crop, ary_pts_crop, c_crop = imgutils.crop_norescale(img_origin, ele_anno)
        img_out, pts_out, c_out = imgutils.change_resolu(img_crop, ary_pts_crop, c_crop, res_heatmap)
        heatmaps = imgutils.generate_heatmaps(img_out, pts_out, sigma_valu=2)
        # plt.figure(1)
        # plt.imshow(img_out)
        # plt.show()
        train_heatmaps[j - 1, ...] = heatmaps
        # plt.figure(1)
        # plt.imshow(train_heatmaps[j-1, :, :, 0])
        # plt.show()
        # imgutils.show_stack_joints(img_out, pts_out, c_out)
        # print('heatmaps.SHAPE', train_heatmaps[j - 1, ...].shape)
        # imgutils.show_heatmaps(img_out, train_heatmaps[j - 1, ...], c_out)
        print('1000 /', j)


# Save train_imgs
name_file = 'np__train_heatmaps_'+str(num_subset)
np.save(name_file, train_heatmaps)
print('=====================================DONE=========================================')