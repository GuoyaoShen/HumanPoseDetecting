import tensorflow as tf
import tensorflow.keras as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datatorch

import numpy as np
import matplotlib.pyplot as plt
import json
import skimage

from utils import imgutils
from utils import imgutilsTF
from utils import Dataset_Make_Utils as dsmutils
from utils import modelutils
from models.hourglass_new import Residual, HourGlass, HourGlassNet, hg
from models.hourglass import hg as hg_torch
from losses.jointMSE import mse_joint as jMSE
from datasets.dataset_torch import DatasetTorch

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#========================================================== TEST BY NEW IMGS ==========================================================
# Load test img
num_img = 6
path_testimg = 'test_imgs/'+str(num_img)+'.jpg'
img_np = imgutils.load_image(path_testimg)
print('img_np.SAPE', img_np.shape)

# Resize to (256,256,3)
img_np = skimage.transform.resize(img_np, [256,256])
img_np_copy = img_np
print('img_np.SAPE', img_np.shape)


# plt.figure(1)
# plt.imshow(img_np_copy)
# plt.show()

# # ================================== Load the model ==================================
# num_subset = 'allEPOCH2'
# net_hg_torch = hg_torch(num_stacks=8, num_blocks=1, num_classes=16)
# path_model_torch_load = 'models/modelparams_hg_torch_' + str(num_subset) + '.pkl'
# # net_hg_torch.load_state_dict(torch.load(path_model_torch_load), strict=False)
# net_hg_torch.load_state_dict(torch.load(path_model_torch_load))
# net_hg_torch.eval()
# print('===============MODEL LOADED===============')

# ================================== Load the checkpoint ==================================
suffix = 'EPOCH5STEP600'  # saved suffix to load  # 4，900   5，300
path_ckpt_torch = 'models/ckpt_hg_torch_' + str(suffix) + '.tar'
checkpoint = torch.load(path_ckpt_torch)
print('===============CHECKPOINT LOADED===============')

net_hg_torch = hg_torch(num_stacks=8, num_blocks=1, num_classes=16)
net_hg_torch.load_state_dict(checkpoint['model_state_dict'])
print('Reconstruct Model DONE')

net_hg_torch.eval()



# ================================== Get heatmaps ==================================
# Reshape and change to Tensor
img_np = np.swapaxes(np.swapaxes(img_np, 0, 2), 1, 2)
img_np = img_np[np.newaxis, ...]  #(1,3,256,256)
print('img_np.HAPE', img_np.shape)
img = torch.from_numpy(img_np).float()

# Get predict heatmap
heatmaps_pred_eg = net_hg_torch(img)
heatmaps_pred_eg = heatmaps_pred_eg
heatmaps_pred_eg = heatmaps_pred_eg[-1].double()  #(1,16,64,64)

# Reshape pred heatmaps
heatmaps_pred_eg_np = heatmaps_pred_eg.detach().numpy()
heatmaps_pred_eg_np = np.squeeze(heatmaps_pred_eg_np, axis=0)
heatmaps_pred_eg_np = np.swapaxes(np.swapaxes(heatmaps_pred_eg_np, 0, 2), 0, 1)  #(64,64,16)

# Show heatmaps
imgutils.show_heatmaps(img_np_copy, heatmaps_pred_eg_np)

# Stack points
coord_joints = modelutils.heatmaps_to_coords(heatmaps_pred_eg_np, resolu_out=[256,256], prob_threshold=0.1)
imgutils.show_stack_joints(img_np_copy, coord_joints, draw_lines=True)



#========================================================== TEST BY DATASET ==========================================================
# num_subset = 1  # 1-22
# num_instance = 277 # 0-999   256
#
# name_file_img = 'datasets/np__train_imgs_' + str(num_subset) + '.npy'
# name_file_heatmap = 'datasets/np__train_heatmaps_' + str(num_subset) + '.npy'
# name_file_pt = 'datasets/np__train_pts_' + str(num_subset) + '.npy'
#
# train_imgs = np.load(name_file_img)
# train_img = train_imgs[num_instance, ...]
# print('img.SHAPE', train_img.shape)
#
# train_heatmaps = np.load(name_file_heatmap)
# train_heatmap = train_heatmaps[num_instance, :, :, :]
# print('heatmap.SHAPE', train_heatmap.shape)
#
# train_pts = np.load(name_file_pt)
# train_points = train_pts[num_instance, ...]
# print('pts.SHAPE', train_points.shape)
#
#
# # plt.figure(1)
# # plt.imshow(img_np_copy)
# # plt.show()
#
# # ================================== Load the model ==================================
# num_subset = 'allEPOCH2'
# net_hg_torch = hg_torch(num_stacks=1, num_blocks=1, num_classes=16)
# path_model_torch_load = 'models/modelparams_hg_torch_' + str(num_subset) + '.pkl'
# # net_hg_torch.load_state_dict(torch.load(path_model_torch_load), strict=False)
# net_hg_torch.load_state_dict(torch.load(path_model_torch_load))
# net_hg_torch.eval()
# print('===============MODEL LOADED===============')
#
#
# # ================================== Get heatmaps ==================================
# # Reshape and change to Tensor
# img_np = np.swapaxes(np.swapaxes(train_img, 0, 2), 1, 2)
# img_np = img_np[np.newaxis, ...]  #(1,3,256,256)
# print('img_np.HAPE', img_np.shape)
# img = torch.from_numpy(img_np).float()
#
# # Get predict heatmap
# heatmaps_pred_eg = net_hg_torch(img)
# heatmaps_pred_eg = heatmaps_pred_eg
# heatmaps_pred_eg = heatmaps_pred_eg[-1].double()  #(1,16,64,64)
#
# # Reshape pred heatmaps
# heatmaps_pred_eg_np = heatmaps_pred_eg.detach().numpy()
# heatmaps_pred_eg_np = np.squeeze(heatmaps_pred_eg_np, axis=0)
# heatmaps_pred_eg_np = np.swapaxes(np.swapaxes(heatmaps_pred_eg_np, 0, 2), 0, 1)  #(64,64,16)
#
# # Show pred heatmaps
# print('=================Pred Heatmaps=================')
# imgutils.show_heatmaps(train_img, heatmaps_pred_eg_np)
# # Stack points
# coord_joints = modelutils.heatmaps_to_coords(heatmaps_pred_eg_np, resolu_out=[256,256], prob_threshold=0.0)
# imgutils.show_stack_joints(train_img, coord_joints)
#
# # Show real heatmaps
# print('=================Real Heatmaps=================')
# imgutils.show_heatmaps(train_img, train_heatmap)