import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import skimage
import json

def generate_heatmap(heatmap, pt, sigma=(33,33), sigma_valu=7):
    # -heatmap: should be a np zeros array with shape (H,W) (only i channel), not (H,W,1)
    # -sigma: should be a tuple with odd values
    # -return: a np array of one joint heatmap with shape (H,W)
    # print('inner heatmap shape', heatmap.shape)
    heatmap[int(pt[1])-1][int(pt[0])-1] = 1
    # heatmap = cv2.GaussianBlur(heatmap, sigma, 0)  #(H,W,1) -> (H,W)
    heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
    am = np.amax(heatmap)
    heatmap = heatmap/am
    return heatmap

def generate_heatmaps(img, pts, sigma=(33,33), sigma_valu=7):
    H, W = img.shape[0], img.shape[1]
    num_pts = pts.shape[0]
    heatmaps = np.zeros((H, W, num_pts))
    # print('heatmap_img.SHAPE', img.shape)
    # print('pts', pts)
    for i, pt in enumerate(pts):
        # print('pt', pt)
        # Filter some points out of the image
        if pt[0] > W:
            pt[0] = W
        if pt[1] > H:
            pt[1] = H
        heatmap = heatmaps[:,:,i]
        heatmap[int(pt[1])-1][int(pt[0])-1] = 1
        # heatmap = cv2.GaussianBlur(heatmap, sigma, 0)  #(H,W,1) -> (H,W)
        heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
        am = np.amax(heatmap)
        heatmap = heatmap / am
        heatmaps[:, :, i] = heatmap
    return heatmaps

def load_image(path_image):
    img = mpimg.imread(path_image)
    # img = cv2.imread(path_image)
    # img = np.array(img)
    return img  # Return a np array (H,W,C)

def crop(img, ele_anno):
    # -img: np array of the origin image
    # -ele_anno: one element of json annotation
    H, W = img.shape[0], img.shape[1]
    s = ele_anno['scale_provided']
    c = ele_anno['objpos']
    # print('img_origin.SHAPE', img.shape)
    # Adjust center and scale
    if c[0] != -1:
        c[1] = c[1] + 15 * s
        s = s * 1.25
    ary_pts = np.array(ele_anno['joint_self'])  # (16, 3)
    # print('pts_origin', ary_pts)
    ary_pts_temp = ary_pts[np.any(ary_pts > [0, 0, 0], axis=1)]
    W_min = max(np.amin(ary_pts_temp, axis=0)[0] - s*10, 0)
    H_min = max(np.amin(ary_pts_temp, axis=0)[1] - s*10, 0)
    W_max = min(np.amax(ary_pts_temp, axis=0)[0] + s*10, W)
    H_max = min(np.amax(ary_pts_temp, axis=0)[1] + s*10, H)

    # Update joint points and center
    ary_pts_crop = ary_pts - np.array([W_min, H_min, 0])
    ary_pts_crop[np.any(ary_pts_crop < [0, 0, 0], axis=1)] = [0, 0, 0]
    c_crop = c - np.array([W_min, H_min])

    img_crop = img[int(H_min):int(H_max), int(W_min):int(W_max), :]
    return img_crop, ary_pts_crop, c_crop

def crop_norescale(img, ele_anno):
    '''
    :param img: np array of the origin image
    :param ele_anno:
    :return: one element of json annotation
    '''

    H, W = img.shape[0], img.shape[1]
    s = ele_anno['scale_provided']
    c = ele_anno['objpos']

    # Adjust center and scale
    if c[0] != -1:
        c[1] = c[1] + 15 * s
        s = s * 1.25

    ary_pts = np.array(ele_anno['joint_self'])  # (16, 3)
    ary_pts_temp = ary_pts[np.any(ary_pts > [0, 0, 0], axis=1)]

    scale_rand = np.random.uniform(low=1.0, high=3.0)

    W_min = max(np.amin(ary_pts_temp, axis=0)[0] - s * 15 * scale_rand, 0)
    H_min = max(np.amin(ary_pts_temp, axis=0)[1] - s * 15 * scale_rand, 0)
    W_max = min(np.amax(ary_pts_temp, axis=0)[0] + s * 15 * scale_rand, W)
    H_max = min(np.amax(ary_pts_temp, axis=0)[1] + s * 15 * scale_rand, H)

    W_len = W_max - W_min
    H_len = H_max - H_min
    window_len = max(H_len, W_len)

    pad_updown = (window_len - H_len)/2
    pad_leftright = (window_len - W_len)/2

    # Calculate 4 corner position
    W_low = max((W_min - pad_leftright), 0)
    W_high = min((W_max + pad_leftright), W)
    H_low = max((H_min - pad_updown), 0)
    H_high = min((H_max + pad_updown), H)

    # Update joint points and center
    ary_pts_crop = ary_pts - np.array([W_low, H_low, 0])
    c_crop = c - np.array([W_low, H_low])

    img_crop = img[int(H_low):int(H_high), int(W_low):int(W_high), :]
    # print('img_crop.SHAPE', img_crop.shape)


    # Pad when H, W different
    H_new, W_new = img_crop.shape[0], img_crop.shape[1]
    window_len_new = max(H_new, W_new)
    pad_updown_new = int((window_len_new - H_new)/2)
    pad_leftright_new = int((window_len_new - W_new)/2)

    # ReUpdate joint points and center (because of the padding)
    ary_pts_crop = ary_pts_crop + np.array([pad_leftright_new, pad_updown_new, 0])
    ary_pts_crop[np.any(ary_pts_crop < [0, 0, 0], axis=1)] = [0, 0, 0]
    c_crop = c_crop + np.array([pad_leftright_new, pad_updown_new])

    img_crop = cv2.copyMakeBorder(img_crop, pad_updown_new, pad_updown_new, pad_leftright_new, pad_leftright_new, cv2.BORDER_CONSTANT, value=0)

    return img_crop, ary_pts_crop, c_crop

def change_resolu(img, pts, c, resolu_out=(256,256)):
    # -img: np array of the origin image
    # -pts: joint points np array corresponding to the image
    # -c: center position
    # -resolu_out: output resolution of the image, should be a numpy array or a tuple

    # RETURN: the pts_out position is calculated according to resolu_out,
    # img_out is also the shape of resolu_out
    # print('img_crop.SHAPE', img.shape)
    H_in = img.shape[0]
    W_in = img.shape[1]
    H_out = resolu_out[0]
    W_out = resolu_out[1]
    H_scale = H_in/H_out
    W_scale = W_in/W_out
    # print('H_scale', H_scale)
    # print('W_scale', W_scale)
    pts_out = pts/np.array([W_scale, H_scale, 1])
    c_out = c/np.array([W_scale, H_scale])
    img_out = skimage.transform.resize(img, tuple(resolu_out))
    # print('img_out.SHAPE', img_out.shape)
    return img_out, pts_out, c_out

def random_rotate(img, resolu_out=(256,256)):
    # -img: np array (H,W,C)
    angle = np.random.random()*360
    img_rot = skimage.transform.rotate(img, angle, resize=True)
    img_rot = skimage.transform.resize(img_rot, resolu_out)
    return img_rot

def show_stack_joints(img, pts, c=[0, 0]):
    # -img: np array (H,W,C)
    # -pts: joint points, np array (16,3)
    # -c: center, np array (2,)

    # In case pts is not np array
    # if type(pts) is np.ndarray:
    pts = np.array(pts)
    # print('pts', pts)

    plt.figure(1)
    plt.imshow(img)
    list_pt_H, list_pt_W = [], []
    list_pt_cH, list_pt_cW = [], []
    for i in range(pts.shape[0]):
        list_pt_W.append(pts[i, 0])  # x axis
        list_pt_H.append(pts[i, 1])  # y axis
    list_pt_cW.append(c[0])
    list_pt_cH.append(c[1])
    plt.scatter(list_pt_W, list_pt_H, color='r', marker='o')
    plt.scatter(list_pt_cW, list_pt_cH, color='b', marker='*')
    plt.axis('off')
    plt.show()
    pass

def show_heatmaps(img, heatmaps, c=np.zeros((2))):
    # -img: np array (H,W,3)
    # -heatmaps: np array (H,W,num_pts)
    # -c: center, np array (2,)
    H, W = img.shape[0], img.shape[1]

    if heatmaps.shape[0] != H:
        heatmaps = skimage.transform.resize(heatmaps, (H, W))

    heatmap_c = generate_heatmap(np.zeros((H, W)), c, (91, 91))
    plt.figure(1)
    for i in range(heatmaps.shape[2] + 1):
        plt.subplot(4, 5, i + 1)
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(heatmaps[:, :, i - 1])
        plt.axis('off')
    plt.subplot(4, 5, 20)
    plt.imshow(heatmap_c)  # Only take in (H,W) or (H,W,3)
    plt.axis('off')
    plt.show()
    pass