'''
Coding by GuoyaoShen
https://github.com/GuoyaoShen/HumanPoseDetecting
'''
import numpy as np
import torch
import torch.utils.data as datatorch
import skimage
import matplotlib.pyplot as plt
import cv2
import json

from utils import imgutils
from utils import modelutils

import os

class DatasetTorch(datatorch.Dataset):
    def __init__(self, is_train=True, use_flip=False, use_rand_scale=False, use_rand_color=False, sigma=2):
        self.is_trian = is_train
        self.use_flip = use_flip
        self.use_rand_scale = use_rand_scale
        self.use_rand_color = use_rand_color
        self.sigma = sigma
        # Get json annotation
        with open('mpii_annotations.json') as anno_file:
            self.anno = json.load(anno_file)  # len 25204

        self.train_list, self.valid_list = [], []  # 22246, 2958
        for idex, ele_anno in enumerate(self.anno):
            if ele_anno['isValidation'] == True:
                self.valid_list.append(idex)
            else:
                self.train_list.append(idex)

    def __getitem__(self, idx):
        if self.is_trian:
            ele_anno = self.anno[self.train_list[idx]]
        else:
            ele_anno = self.anno[self.valid_list[idx]]
        res_img = [256, 256]
        res_heatmap = [64, 64]
        path_img_folder = 'mpii_human_pose_v1/images'
        path_img = os.path.join(path_img_folder, ele_anno['img_paths'])
        img_origin = imgutils.load_image(path_img)
        img_crop, ary_pts_crop, c_crop = imgutils.crop_norescale(img_origin, ele_anno)
        img_out, pts_out, c_out = imgutils.change_resolu(img_crop, ary_pts_crop, c_crop, res_heatmap)

        train_img = skimage.transform.resize(img_crop, tuple(res_img))
        train_heatmap = imgutils.generate_heatmaps(img_out, pts_out, sigma_valu=self.sigma)
        train_pts = pts_out[:, :2].astype(np.int32)

        # (H,W,C) -> (C,H,W)
        train_img = np.transpose(train_img, (2, 0, 1))
        train_heatmap = np.transpose(train_heatmap, (2, 0, 1))

        if self.use_flip:
            train_img, train_heatmap, train_pts = random_flip_LR(train_img, train_heatmap, train_pts)
        if self.use_rand_scale:
            train_img, train_heatmap, train_pts = random_scale(train_img, train_heatmap, train_pts)
        if self.use_rand_color:
            train_img = rand_color(train_img)
        return train_img, train_heatmap, train_pts

    def __len__(self):
        if self.is_trian:
            return len(self.train_list)
        else:
            return len(self.valid_list)

def random_flip_LR(img, heatmaps, pts):
    '''
    :param img: only 1 img, shape(3,256,256), np array
    :param heatmaps: shape(16,64,64), np array
    :param pts: shape(16,2), np array, same resolu as heatmaps
    :return: same as input

    When flip, the order of heatmaps also need to change.
    '''
    H_img, W_img = img.shape[1], img.shape[2]
    H_hm, W_hm = heatmaps.shape[1], heatmaps.shape[2]
    flip = np.random.random() > 0.5
    # flip = True
    # print('FLIP:', flip)
    if not flip:
        return img, heatmaps, pts
    #! pytorch not supported negative stride for now, use copy()
    # (C,H,W)
    img = np.flip(img, 2).copy()
    heatmaps = np.flip(heatmaps, 2).copy()
    # Rearrange heatmaps
    heatmaps = np.concatenate((heatmaps[5::-1], heatmaps[6:10], heatmaps[15:9:-1])).copy()

    # Calculate flip pts, remember to filter [0,0] which is no available heatmap
    pts = np.where(pts == [0, 0], pts, [W_hm, 0] + pts*[-1, 1])

    # Rearrange pts
    pts = np.concatenate((pts[5::-1], pts[6:10], pts[15:9:-1])).copy()
    return img, heatmaps, pts

def rand_color(img):
    '''
    :param img: only 1 img, shape(3,256,256), np array
    :return: shape(3,256,256), np array, with color rand changed
    '''
    img[0] *= np.clip(np.random.uniform(low=0.8, high=1.2), 0., 1.)
    img[1] *= np.clip(np.random.uniform(low=0.8, high=1.2), 0., 1.)
    img[2] *= np.clip(np.random.uniform(low=0.8, high=1.2), 0., 1.)
    return img

def random_scale(img, heatmaps, pts):
    '''
    :param img: only 1 img, shape(3,256,256), np array
    :param heatmaps: shape(16,64,64), np array
    :param pts: shape(16,2), np array, same resolu as heatmaps
    :return: same as input

    Obsolete, rand scale now in the crop function
    '''

    scale = np.random.uniform(low=0.5, high=1.0)
    H_img, W_img = img.shape[1], img.shape[2]
    H_hm, W_hm = heatmaps.shape[1], heatmaps.shape[2]

    # (C,H,W) -> (H,W,C)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    heatmaps = np.swapaxes(np.swapaxes(heatmaps, 0, 2), 0, 1)

    img = skimage.transform.rescale(img, [scale, scale], multichannel=3)
    heatmaps = skimage.transform.rescale(heatmaps, [scale, scale], multichannel=16)
    H_img_new, W_img_new = img.shape[0], img.shape[1]
    H_hm_new, W_hm_new = heatmaps.shape[0], heatmaps.shape[1]
    pad_leftright_img, pad_updown_img = int((W_img - W_img_new)/2), int((H_img - H_img_new)/2)
    pad_leftright_hm, pad_updown_hm = int((W_hm - W_hm_new)/2), int((H_hm - H_hm_new)/2)
    img = cv2.copyMakeBorder(img, pad_updown_img, pad_updown_img, pad_leftright_img, pad_leftright_img,
                                  cv2.BORDER_CONSTANT, value=0)
    heatmaps = cv2.copyMakeBorder(heatmaps, pad_updown_hm, pad_updown_hm, pad_leftright_hm, pad_leftright_hm,
                                  cv2.BORDER_CONSTANT, value=0)
    # resize again to guarantee (256,256) & (64,64)
    img = skimage.transform.resize(img, tuple([H_img, W_img]))
    heatmaps = skimage.transform.resize(heatmaps, tuple([H_hm, W_hm]))

    # (H,W,C) -> (C,H,W)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    heatmaps = np.swapaxes(np.swapaxes(heatmaps, 0, 2), 1, 2)

    # Calculate flip pts
    pts = pts*scale + [pad_leftright_hm, pad_updown_hm]

    return img, heatmaps, pts


#==================== Test ====================
if __name__ == '__main__':
    pass