import numpy as np
import torch
import torch.utils.data as datatorch
import skimage
import matplotlib.pyplot as plt
import cv2
import json

from utils import imgutils
from utils import modelutils
from datasets.dataset_torch_new import DatasetTorch

import os

#==================== Test ====================
# if __name__ == '__main__':
#     # ==================== Test of dataset====================
#     ds_torch = DatasetTorch()
#     data_loader = datatorch.DataLoader(ds_torch, batch_size=1, shuffle=False)
#     for step, (img, heatmaps, pts) in enumerate(data_loader):
#         print('img is TENSOR', torch.is_tensor(img))  # True
#         print('heatmaps is TENSOR', torch.is_tensor(heatmaps))  # True
#         # print('img', img)
#         # print('heatmaps', heatmaps)
#         print('img.SHAPE', img.shape)  # [1,3,256,256]
#         print('heatmaps.SHAPE', heatmaps.shape)  # [1,16,64,64]
#
#         # ====== Show img and heatmaps ======
#         img_np = torch.squeeze(img, 0).numpy()  # (C,H,W)
#         heatmaps_np = torch.squeeze(heatmaps, 0).numpy()
#         # print('img_np', img_np.shape)
#         # print('heatmaps_np', heatmaps_np.shape)
#         img_np = np.swapaxes(np.swapaxes(img_np, 0, 2), 0, 1)
#         heatmaps_np = np.swapaxes(np.swapaxes(heatmaps_np, 0, 2), 0, 1)
#         print('img_np', img_np.shape)
#         print('heatmaps_np', heatmaps_np.shape)
#
#         imgutils.show_heatmaps(img_np, heatmaps_np)
#         joint_pts = modelutils.heatmaps_to_coords(heatmaps_np, resolu_out=[64,64], prob_threshold=0.0)
#         img_np = skimage.transform.resize(img_np, [64,64])
#         imgutils.show_stack_joints(img_np, joint_pts, c=[0, 0])
#         pass

if __name__ == '__main__':
    # ========== Test dataset ==========
    ds_torch = DatasetTorch(use_flip=True, use_rand_color=True)
    data_loader = datatorch.DataLoader(ds_torch, batch_size=1, shuffle=False)
    for step, (img, heatmaps, pts) in enumerate(data_loader):
        print('img', img)
        # print('heatmaps', heatmaps.shape)
        # print('pts', pts.shape)

        img = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))
        # img = np.fliplr(img)
        heatmaps = np.transpose(heatmaps.squeeze().detach().numpy(), (1, 2, 0))
        pts = pts.squeeze().detach().numpy()
        print('pts', pts)
        print('===========================================================')
        imgutils.show_heatmaps(img, heatmaps)

