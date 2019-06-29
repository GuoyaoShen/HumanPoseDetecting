import tensorflow as tf
import tensorflow.keras as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datatorch

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math

from utils import imgutils
from utils import imgutilsTF
from utils import Dataset_Make_Utils as dsmutils
from utils import modelutils
from models.hourglass_new import Residual, HourGlass, HourGlassNet, hg
from models.hourglass import hg as hg_torch
from losses.jointMSE import mse_joint as jMSE
from losses.jointsmseloss import JointsMSELoss
from datasets.dataset_torch_new import DatasetTorch

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




# ================================== Load data ==================================
num_subset = 1



# ================================== Construct dataset ==================================
ds_torch = DatasetTorch()
data_loader = datatorch.DataLoader(ds_torch, batch_size=32, shuffle=True)


# ================================== Construct model ==================================
# device = torch.device('cuda:0')
device = torch.device('cpu')
learning_rate = 1e-3

net_hg_torch = hg_torch(num_stacks=1, num_blocks=1, num_classes=16).to(device)
optimizer = torch.optim.RMSprop(net_hg_torch.parameters(), lr=learning_rate)
# criteon = torch.nn.MSELoss().to(device)
criteon = JointsMSELoss(use_target_weight=True).to(device)


# ================================== Train ==================================
num_epoch = 10
num_setsize = ds_torch.__len__()
print('num_setsize', num_setsize)
target_weight = torch.ones(1, 16)

plt.ion()
for i in range(num_epoch):
    for step, (img, heatmaps, pts) in enumerate(data_loader):
        # To GPU
        # img, heatmaps = img.to(device), heatmaps.cuda()

        # All dtype change to
        img, heatmaps = img.float(), heatmaps.float()


        print('')
        print('EPOCH', str(num_epoch), '/', i+1, ' ||  STEP', math.ceil(num_setsize/32), '/', step + 1)
        heatmaps_pred = net_hg_torch(img)
        heatmaps_pred = heatmaps_pred[-1]

        heatmaps_pred_copy = heatmaps_pred[1]
        img_copy = img[1]

        loss = criteon(heatmaps_pred, heatmaps, target_weight)
        print('LOSS:', loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Show pred heatmaps
        heatmaps_pred_np = heatmaps_pred_copy.detach().numpy()
        # heatmaps_pred_np = np.squeeze(heatmaps_pred_np, axis=0)
        heatmaps_pred_np = np.swapaxes(np.swapaxes(heatmaps_pred_np, 0, 2), 0, 1)  # (64,64,16)
        img_np = img_copy.detach().numpy()
        # img_np = np.squeeze(img_np, axis=0)
        img_np = np.swapaxes(np.swapaxes(img_np, 0, 2), 0, 1)  # (64,64,16)
        imgutils.show_heatmaps(img_np, heatmaps_pred_np)
        plt.pause(1)
plt.ioff()



        # if step > 5:
        #     break




# ================================== Load the model ==================================
# net_hg_torch = hg_torch(num_stacks=1, num_blocks=1, num_classes=16)
# path_model_torch_load = 'models/modelparams_hg_torch_' + str(num_subset) + '.pkl'
# net_hg_torch.load_state_dict(torch.load(path_model_torch_load))
# net_hg_torch.eval()
# print('===============MODEL LOADED===============')

# ================================== Save the model ==================================
path_model_torch = 'models/modelparams_hg_torch_' + str(num_subset) + '.pkl'
torch.save(net_hg_torch.state_dict(), path_model_torch)
print('===============MODEL PARAMS SAVED===============')

# ================================== Visualize ==================================
path_train_img = 'datasets/np__train_imgs_' + str(num_subset) + '.npy'
path_train_heatmap = 'datasets/np__train_heatmaps_' + str(num_subset) + '.npy'
path_train_pt = 'datasets/np__train_pts_' + str(num_subset) + '.npy'

train_imgs_np = np.load(path_train_img)  #(1000,256,256,3)
train_heatmaps_np = np.load(path_train_heatmap)  #(1000,64,64,16)
train_pts_np = np.load(path_train_pt)  #(1000,16,2)

# Load a single image
train_img_np = train_imgs_np[num_subset, ...]  #(256,256,3)
train_heatmap_np = train_heatmaps_np[num_subset, ...]   #(64,64,16)


# Reshape and change to Tensor
train_img = np.swapaxes(np.swapaxes(train_img_np, 0, 2), 1, 2)
train_img = train_img[np.newaxis, ...]  #(1,3,256,256)
train_img = torch.from_numpy(train_img).float()

# Get predict heatmap
heatmaps_pred_eg = net_hg_torch(train_img)
heatmaps_pred_eg = heatmaps_pred_eg
heatmaps_pred_eg = heatmaps_pred_eg[-1].double()  #(1,16,64,64)

# Reshape pred heatmaps
heatmaps_pred_eg_np = heatmaps_pred_eg.detach().numpy()
heatmaps_pred_eg_np = np.squeeze(heatmaps_pred_eg_np, axis=0)
heatmaps_pred_eg_np = np.swapaxes(np.swapaxes(heatmaps_pred_eg_np, 0, 2), 0, 1)  #(64,64,16)




# Show imgs
# imgutils.show_stack_joints(train_img_np, coord_joints_true)
# imgutils.show_stack_joints(train_img_np, coord_joints_eg_pred)

# Show heatmaps
imgutils.show_heatmaps(train_img_np, train_heatmap_np)
imgutils.show_heatmaps(train_img_np, heatmaps_pred_eg_np)


print('===============================DONE===============================')