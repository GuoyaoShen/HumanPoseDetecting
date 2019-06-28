import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from utils import imgutils
from utils import imgutilsTF
from utils import Dataset_Make_Utils as dsmutils
from utils import modelutils
from models.hourglass_new import Residual, HourGlass, HourGlassNet, hg
from losses.jointMSE import mse_joint as jMSE

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# tf.debugging.set_log_device_placement(True)



# ================================== Load data ==================================
num_subset = 1
name_file_img = 'datasets/np__train_imgs_'+str(num_subset)+'.npy'
name_file_heatmap = 'datasets/np__train_heatmaps_'+str(num_subset)+'.npy'
name_file_pt = 'datasets/np__train_pts_'+str(num_subset)+'.npy'

train_imgs = tf.convert_to_tensor(np.load(name_file_img))  #(1000,256,256,3)
train_imgs = tf.cast(train_imgs, tf.float32)
# print('train_imgs TENSOR', tf.is_tensor(train_imgs))
# print('train_imgs.SHAPE', train_imgs.shape)

train_heatmaps = tf.convert_to_tensor(np.load(name_file_heatmap))  #(1000,64,64,16)
train_heatmaps = tf.cast(train_heatmaps, tf.float32)
# print('train_heatmaps TENSOR', tf.is_tensor(train_heatmaps))
# print('train_heatmaps.SHAPE', train_heatmaps.shape)

train_pts = tf.convert_to_tensor(np.load(name_file_pt))  #(1000,16,2)
train_pts = tf.cast(train_pts, tf.float32)
# print('train_pts TENSOR', tf.is_tensor(train_pts))
# print('train_pts.SHAPE', train_pts.shape)
# print(train_pts)
# ================================== Load data ==================================





#********************************************************************************

# ================================== Construct dataset ==================================
train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_heatmaps)).shuffle(10000).batch(32)
# ================================== Construct dataset ==================================

#********************************************************************************

# ================================== Construct model ==================================
net_hg = hg(num_stack=2, num_block=1, num_class=16)
# ================================== Construct model ==================================

#********************************************************************************

# ================================== Train ==================================
optimizer = K.optimizers.RMSprop(learning_rate=1e-3)
loss_mse = K.losses.MeanSquaredError()
num_epoch = 1
for i in range(num_epoch):
    for step, (img, heatmap) in enumerate(train_ds):
        print('')
        print('EPOCH 5 /', i+1, ' ||  RANGE 32 /', step + 1)
        # print('TRAIN IMG SHAPE', img.shape)  #(32,256,256,3)
        # print('TRAIN PTS SHAPE', pts.shape)  #(32,16,2)
        with tf.GradientTape(persistent=True) as tape:
            list_loss = []
            heatmap_pred = net_hg(img)  # dtype=float32
            for n in range(len(heatmap_pred)):
                loss_n = jMSE(heatmap, heatmap_pred[n])
                list_loss.append(loss_n)
            # heatmap_pred = heatmap_pred[-1]
            # loss = jMSE(heatmap, heatmap_pred)
            print('LOSS', list_loss)
        for n in range(len(heatmap_pred)):
            loss_n = list_loss[n]
            gradients = tape.gradient(loss_n, net_hg.trainable_variables)  # len 218
            optimizer.apply_gradients(zip(gradients, net_hg.trainable_variables))
        del tape
        # gradients = tape.gradient(list_loss, net_hg.trainable_variables)  # len 218
        # optimizer.apply_gradients(zip(gradients, net_hg.trainable_variables))

# for step, (img, heatmap) in enumerate(train_ds):
#     print('32 /', step+1)
#     # print('TRAIN IMG SHAPE', img.shape)  #(32,256,256,3)
#     # print('TRAIN PTS SHAPE', pts.shape)  #(32,16,2)
#     with tf.GradientTape() as tape:
#         heatmap_pred = net_hg(img)  # dtype=float32
#         heatmap_pred = heatmap_pred[0]
#         loss = jMSE(heatmap, heatmap_pred)
#         print('LOSS', loss)
#     gradients = tape.gradient(loss, net_hg.trainable_variables)  # len 218
#     optimizer.apply_gradients(zip(gradients, net_hg.trainable_variables))
# ================================== Train ==================================

#********************************************************************************

# ================================== Save the model ==================================
# name_file_model = 'models/hourglass_save_'+str(num_subset)
# # net_hg.save(name_file_model, save_format='tf')
# K.models.save_model(net_hg, name_file_model, save_format='tf')
# ================================== Save the model ==================================

#********************************************************************************

# ================================== Visualize ==================================
# Load a single image
num_eg = 123
img_eg = train_imgs[num_eg, ...]  #(256,256,3)
img_eg_np = img_eg.numpy()
img_eg = img_eg[np.newaxis, ...]  #(1,256,256,3)
# Load heatmaps
heatmaps_eg = train_heatmaps[num_eg, ...]
heatmaps_eg = heatmaps_eg.numpy()



heatmaps_eg_pred = heatmap_pred = net_hg(img_eg)
heatmaps_eg_pred = heatmaps_eg_pred[-1]

heatmaps_eg_pred = tf.squeeze(heatmaps_eg_pred, axis=0)  #(64,64,16)
heatmaps_eg_pred_np = heatmaps_eg_pred.numpy()



coord_joints_eg_pred = modelutils.heatmaps_to_coords(heatmaps_eg_pred, resolu_out=[256, 256])
coord_joints_true = train_pts[num_eg, ...].numpy()*(256/64)

# Show imgs
imgutils.show_stack_joints(img_eg_np, coord_joints_true)
imgutils.show_stack_joints(img_eg_np, coord_joints_eg_pred)

# Show heatmaps
imgutils.show_heatmaps(img_eg_np, heatmaps_eg)
imgutils.show_heatmaps(img_eg_np, heatmaps_eg_pred_np)

# ================================== Visualize ==================================

print('===============================DONE===============================')