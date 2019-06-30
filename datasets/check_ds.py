import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import imgutils
import skimage
import os

from utils import imgutils
from utils import imgutilsTF
from utils import Dataset_Make_Utils as dsmutils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#============== Check img dataset ================
# num_subset = 10
# name_file = 'np__train_imgs_'+str(num_subset)+'.npy'
# train_imgs = np.load(name_file)
# img = train_imgs[999, ...]
# plt.figure(1)
# plt.imshow(img)
# plt.show()
#============== Check img dataset ================



#============== Check heatmap dataset ================
# num_subset = 1
# name_file = 'np__train_heatmaps_'+str(num_subset)+'.npy'
# train_heatmaps = np.load(name_file)
# heatmap = train_heatmaps[999, :, :, 16]
# plt.figure(1)
# plt.imshow(heatmap)
# plt.show()
#============== Check img dataset ================


#============== Combine images and heatmaps and joint points ================
num_subset = 5  # 1-22
num_instance = 356  # 0-999

name_file_img = 'np__train_imgs_'+str(num_subset)+'.npy'
name_file_heatmap = 'np__train_heatmaps_'+str(num_subset)+'.npy'
name_file_pt = 'np__train_pts_'+str(num_subset)+'.npy'

train_imgs = np.load(name_file_img)
img = train_imgs[num_instance, ...]
print('img.SHAPE', img.shape)

train_heatmaps = np.load(name_file_heatmap)
heatmap = train_heatmaps[num_instance, :, :, :]
print('heatmap.SHAPE', heatmap.shape)

train_pts = np.load(name_file_pt)
pts = train_pts[num_instance, ...]
print('pts.SHAPE', pts.shape)

# Show heatmaps
# imgutils.show_heatmaps(img, heatmap)

# Show stack joint points
img_low = skimage.transform.resize(img, (64,64))
imgutils.show_stack_joints(img_low, pts)
#============== Combine images and heatmaps ================