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

res_img = [256, 256]
res_heatmap = [64, 64]
num_heatmap = 16


# Get parent dir path
path_dir = os.getcwd()
path_dir = os.path.dirname(path_dir)


idx_begin = 8000
idx_end = 11999
len_idx = (idx_end - idx_begin + 1)

train_imgs = np.zeros((len_idx, res_img[0], res_img[1], 3))  #(N,H_im,W_im,3)
path_img_folder = path_dir+'/mpii_human_pose_v1/images'

num_subset = '4000(3)'  # 0 means all


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
        img_crop, ary_pts_crop, c_crop = imgutils.crop_norescale(img_origin, ele_anno, use_randscale=False)
        img_out, pts_out, c_out = imgutils.change_resolu(img_crop, ary_pts_crop, c_crop, res_img)
        # plt.figure(1)
        # plt.imshow(img_out)
        # plt.show()
        train_imgs[j-1, ...] = img_out
        # plt.figure(1)
        # plt.imshow(train_imgs[i])
        # plt.show()
        print(len_idx, '/', j)


# Save train_imgs
print('LEN train img', train_imgs.shape[0])
name_file = 'np__train_imgs_'+str(num_subset)
np.save(name_file, train_imgs)
print('=====================================DONE=========================================')