import numpy as np
import torch
import torch.utils.data as datatorch
import skimage
import matplotlib.pyplot as plt
import cv2

from utils import imgutils
from utils import modelutils

class DatasetTorch(datatorch.Dataset):
    def __init__(self, num_subset=1, use_flip=True, use_rand_scale=True):
        self.num_subset = num_subset
        self.use_flip = use_flip
        self.use_rand_scale = use_rand_scale
        path_train_img = 'datasets/np__train_imgs_' + str(num_subset) + '.npy'
        path_train_heatmap = 'datasets/np__train_heatmaps_' + str(num_subset) + '.npy'
        path_train_pt = 'datasets/np__train_pts_' + str(num_subset) + '.npy'

        # These paths used for start in this script
        # path_train_img = 'np__train_imgs_' + str(num_subset) + '.npy'
        # path_train_heatmap = 'np__train_heatmaps_' + str(num_subset) + '.npy'
        # path_train_pt = 'np__train_pts_' + str(num_subset) + '.npy'

        # Load data
        train_imgs = np.load(path_train_img)  # (1000,256,256,3)
        train_heatmaps = np.load(path_train_heatmap)  #(1000,64,64,16)
        self.train_points = np.load(path_train_pt)  # (1000,16,2)
        self.len = train_imgs.shape[0]

        # Change axis to (N,C,H,W)
        self.train_imgs = np.swapaxes(np.swapaxes(train_imgs, 1, 3), 2, 3)
        self.train_heatmaps = np.swapaxes(np.swapaxes(train_heatmaps, 1, 3), 2, 3)

        pass

    def __getitem__(self, idx):
        train_img = self.train_imgs[idx]  #(3,256,256)
        train_heatmap = self.train_heatmaps[idx]  #(16,64,64)
        train_pts = self.train_points[idx]  #(16,2)
        if self.use_flip:
            train_img, train_heatmap, train_pts = random_flip_LR(train_img, train_heatmap, train_pts)
        if self.use_rand_scale:
            train_img, train_heatmap, train_pts = random_scale(train_img, train_heatmap, train_pts)
        return train_img, train_heatmap, train_pts

    def __len__(self):
        len = self.len
        return len

def random_flip_LR(img, heatmaps, pts):
    '''
    :param img: only 1 img, shape(3,256,256), np array
    :param heatmaps: shape(16,64,64), np array
    :param pts: shape(16,2), np array, same resolu as heatmaps
    :return: same as input
    '''
    H_img, W_img = img.shape[1], img.shape[2]
    H_hm, W_hm = heatmaps.shape[1], heatmaps.shape[2]
    flip = np.random.random() > 0.5
    # print('FLIP:', flip)
    if not flip:
        return img, heatmaps, pts
    #! pytorch not supported negative stride for now
    # img = img[:, :, ::-1]
    # heatmaps = heatmaps[:, :, ::-1]

    # (C,H,W) -> (H,W,C)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    heatmaps = np.swapaxes(np.swapaxes(heatmaps, 0, 2), 0, 1)
    img = cv2.flip(img, 1)
    heatmaps = cv2.flip(heatmaps, 1)
    # (H,W,C) -> (C,H,W)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    heatmaps = np.swapaxes(np.swapaxes(heatmaps, 0, 2), 1, 2)

    # Calculate flip pts
    pts = [W_hm, 0] + pts*[-1, 1]
    return img, heatmaps, pts

def random_scale(img, heatmaps, pts):
    '''
    :param img: only 1 img, shape(3,256,256), np array
    :param heatmaps: shape(16,64,64), np array
    :param pts: shape(16,2), np array, same resolu as heatmaps
    :return: same as input
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
    # ========== Test dataset ==========
    # ds_torch = DatasetTorch()
    # data_loader = datatorch.DataLoader(ds_torch, batch_size=1, shuffle=False)
    # for step, (img, heatmaps) in enumerate(data_loader):
    #     print('img is TENSOR', torch.is_tensor(img))  # True
    #     print('heatmaps is TENSOR', torch.is_tensor(heatmaps))  # True
    #     # print('img', img)
    #     # print('heatmaps', heatmaps)
    #     print('img.SHAPE', img.shape)  #[1,3,256,256]
    #     print('heatmaps.SHAPE', heatmaps.shape)  #[1,16,64,64]
    #
    #     # ====== Show img and heatmaps ======
    #     img_np = torch.squeeze(img, 0).numpy()  #(C,H,W)
    #     heatmaps_np = torch.squeeze(heatmaps, 0).numpy()
    #     # print('img_np', img_np.shape)
    #     # print('heatmaps_np', heatmaps_np.shape)
    #     img_np = np.swapaxes(np.swapaxes(img_np, 0, 2), 0, 1)
    #     heatmaps_np = np.swapaxes(np.swapaxes(heatmaps_np, 0, 2), 0, 1)
    #     print('img_np', img_np.shape)
    #     print('heatmaps_np', heatmaps_np.shape)
    #     imgutils.show_heatmaps(img_np, heatmaps_np)

    # ========== Test Flip LR ==========
    num_subset = 5  # 1-22
    num_instance = 145  # 0-999   256

    name_file_img = 'np__train_imgs_' + str(num_subset) + '.npy'
    name_file_heatmap = 'np__train_heatmaps_' + str(num_subset) + '.npy'
    name_file_pt = 'np__train_pts_' + str(num_subset) + '.npy'

    train_imgs = np.load(name_file_img)
    img = train_imgs[num_instance, ...]
    print('img.SHAPE', img.shape)

    train_heatmaps = np.load(name_file_heatmap)
    heatmap = train_heatmaps[num_instance, :, :, :]
    print('heatmap.SHAPE', heatmap.shape)

    train_pts = np.load(name_file_pt)
    pts = train_pts[num_instance, ...]
    print('pts.SHAPE', pts.shape)

    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    heatmap = np.swapaxes(np.swapaxes(heatmap, 0, 2), 1, 2)

    img, heatmap, pts = random_flip_LR(img, heatmap, pts)

    img = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    heatmap = np.swapaxes(np.swapaxes(heatmap, 0, 2), 0, 1)

    # plt.figure(1)
    # plt.imshow(heatmap)
    # plt.show()

    # Show heatmaps
    imgutils.show_heatmaps(img, heatmap)
    pts_hm = modelutils.heatmaps_to_coords(heatmap, resolu_out=[64,64], prob_threshold=0)
    img_low = skimage.transform.resize(img, (64, 64))
    print('================from HEATMAPS================')
    imgutils.show_stack_joints(img_low, pts_hm)

    # Show stack joint points
    print('================from PTS================')
    img_low = skimage.transform.resize(img, (64, 64))
    imgutils.show_stack_joints(img_low, pts)

    # ========== Test random scale ==========
    # num_subset = 5  # 1-22
    # num_instance = 356  # 0-999
    #
    # name_file_img = 'np__train_imgs_' + str(num_subset) + '.npy'
    # name_file_heatmap = 'np__train_heatmaps_' + str(num_subset) + '.npy'
    # name_file_pt = 'np__train_pts_' + str(num_subset) + '.npy'
    #
    # train_imgs = np.load(name_file_img)
    # img = train_imgs[num_instance, ...]
    # print('img.SHAPE', img.shape)
    #
    # train_heatmaps = np.load(name_file_heatmap)
    # heatmap = train_heatmaps[num_instance, :, :, :]
    # print('heatmap.SHAPE', heatmap.shape)
    #
    # train_pts = np.load(name_file_pt)
    # pts = train_pts[num_instance, ...]
    # print('pts.SHAPE', pts.shape)
    #
    # img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    # heatmap = np.swapaxes(np.swapaxes(heatmap, 0, 2), 1, 2)
    #
    # img, heatmap, pts = random_scale(img, heatmap, pts)
    #
    # img = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    # heatmap = np.swapaxes(np.swapaxes(heatmap, 0, 2), 0, 1)
    #
    # print('SCALE img.SHAPE', img.shape)
    #
    # # plt.figure(1)
    # # plt.imshow(img)
    # # plt.show()
    #
    # # Show heatmaps
    # imgutils.show_heatmaps(img, heatmap)
    # pts_hm = modelutils.heatmaps_to_coords(heatmap, resolu_out=[64,64], prob_threshold=0)
    # img_low = skimage.transform.resize(img, (64, 64))
    # print('================from HEATMAPS================')
    # imgutils.show_stack_joints(img_low, pts_hm)
    #
    # # Show stack joint points
    # print('================from PTS================')
    # img_low = skimage.transform.resize(img, (64, 64))
    # imgutils.show_stack_joints(img_low, pts)