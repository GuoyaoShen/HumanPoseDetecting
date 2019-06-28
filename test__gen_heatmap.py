import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json

from utils import imgutils
from utils import imgutilsTF

# def generate_heatmap(heatmap_origin, pts, sigma):
#     heatmap = heatmap_origin
#     # heatmap = np.zeros((720,1080,3))
#     heatmap.flags.writeable = True
#     # Filter for available points
#     for pt in pts:
#         if pt[0] > 0 or pt[1] > 0:
#             heatmap[int(pt[1])][int(pt[0])] = 1
#     heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
#     am = np.amax(heatmap)
#     # print('am:', am)
#     heatmap = heatmap/am
#     return heatmap
#
#
# with open('mpii_annotations.json') as anno_file:
#     anno = json.load(anno_file)
# # Search for specific photo, get joint points
# print('============================================================================')
# for ele in anno:
#     if ele['img_paths'] == '002310685.jpg':
#         print('FOUNDED')
#         ary_pts = np.array(ele['joint_self'])
#
# print(ary_pts.shape)
# # print(ary_pts)
#
#
# # Get image
# img = mpimg.imread('mpii_human_pose_v1/images/002310685.jpg')
# # array_img = np.array(img)
# array_img = np.zeros((np.array(img).shape[0],np.array(img).shape[1],1))
# # array_img = np.swapaxes(array_img, 0, 2).swapaxes(1, 2)  # (C,H,W)
# # plt.imshow(array_img) # 显示图片
# # plt.axis('off') # 不显示坐标轴
# # plt.show()
#
#
# # filter unavailable points
# # for ary_pt in ary_pts:
# #     # print(ary_pt)
# #     if ary_pt[0]>0 or ary_pt[1]>0:
# #         print(ary_pt)
# #         array_img = generate_heatmap(array_img, ary_pt, (9, 9))
# array_img = generate_heatmap(array_img, ary_pts, (17, 17))
#
# # Show img
# # img = mpimg.imread('mpii_human_pose_v1/images/002310685.jpg')
# # array_img = np.array(img)
# #
# # print(array_img.shape)  #(H,W,C)
# # array_img = np.swapaxes(array_img, 0, 2).swapaxes(1, 2)  # (C,H,W)
# # print('new shape:', array_img.shape)
# #
#
# plt.figure(1)
# plt.subplot(121)
# plt.imshow(img) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.subplot(122)
# plt.imshow(array_img) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()


#==================================================================
# # def generate_heatmap(heatmap, pt, sigma):
# #     # sigma should be a tuple with odd values
# #     # heatmap.flags.writeable = True
# #     heatmap[int(pt[1])][int(pt[0])] = 1
# #     heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
# #     am = np.amax(heatmap)
# #     # print('am:', am)
# #     heatmap = heatmap/am
# #     return heatmap
#
#
# # Get json annotation
# with open('mpii_annotations.json') as anno_file:
#     anno = json.load(anno_file)
# # Search for specific photo, get joint points
# for ele in anno:
#     if ele['img_paths'] == '005123954.jpg':
#         ele_copy = ele
#         print('FOUNDED')
#         ary_pts = np.array(ele['joint_self'])  # (16,3)
#         ary_pts_origin = ary_pts
#         print('objpos:', ele['objpos'])
#         c = ele['objpos']
#         c_origin = c.copy()
#         s = ele['scale_provided']
#         if c[0] != -1:
#             c[1] = c[1] + 15 * s
#             s = s * 1.25
#         print('NEW objpos:', c)
#         print('scale_provided:', ele['scale_provided'])
#         print('NEW scale_provided:', s)
#         print('ary_pts:', ary_pts)
#         num_joints = np.array(ele['joint_self']).shape[0]
#
# # Get image
# # img = imgutils.load_image('mpii_human_pose_v1/images/000033016.jpg')
# img_origin = imgutils.load_image('mpii_human_pose_v1/images/005123954.jpg')
# img, ary_pts_new, c = imgutils.crop(img_origin, ele_copy)
# print('new_joints', ary_pts_new)
# img_lowres, ary_pts ,c = imgutils.change_resolu(img, ary_pts_new, c, np.array([256, 256]))
# # plt.figure(1)
# # # plt.imshow(img)
# # plt.imshow(img_lowres)
# # plt.show()
#
# # H, W, C = np.array(img).shape[0], np.array(img).shape[1], np.array(img).shape[2]
# H, W, C = np.array(img_origin).shape[0], np.array(img_origin).shape[1], np.array(img_origin).shape[2]
#
# # Generate heatmaps
# sigma = (91, 91)
# heatmaps = np.zeros((H, W, num_joints))
# for i, ary_pt in enumerate(ary_pts_origin):
#     print('ary_pt:', i, ary_pt)
#     if ary_pt[0] > 0 or ary_pt[1] > 0:
#         heatmaps[:, :, i] = imgutils.generate_heatmap(heatmaps[:, :, i], ary_pt, sigma)
#         print('heatmaps', i, heatmaps[:, :, i].shape)
# print('HEATMAPS', heatmaps.shape)
# print('c_origin', c_origin)
# img_obj = imgutils.generate_heatmap(np.zeros((H, W)), c_origin, (91,91))
# print(img_obj.shape)
#
# # Show images
# # plt.figure(1)
# # for i in range(num_joints+1):
# #     plt.subplot(4, 5, i+1)
# #     if i == 0:
# #         plt.imshow(img_origin)
# #     else:
# #         plt.imshow(heatmaps[:, :, i-1])
# #     plt.axis('off')
# # plt.subplot(4, 5, 18)
# # plt.imshow(img_obj)  # Only take in (H,W) or (H,W,3)
# # plt.axis('off')
# # plt.show()
# # imgutils.show_heatmaps(img_origin, heatmaps, c_origin)
#
#
# # Show points on image
# imgutils.show_stack_joints(img_lowres, ary_pts, c)
#=======================================================================================
# Get json annotation
with open('mpii_annotations.json') as anno_file:
    anno = json.load(anno_file)
for ele in anno:
    if ele['img_paths'] == '000678817.jpg':  #000033016, 000678817, 001834988
        ele_copy = ele
        print('FOUNDED')
        print(ele)

c_copy = ele_copy['objpos']
pts_copy = ele_copy['joint_self']

img_origin = imgutils.load_image('mpii_human_pose_v1/images/000678817.jpg')
img_copy = img_origin
# img_crop, ary_pts_crop, c_crop = imgutils.crop(img_origin, ele_copy)
img_crop, ary_pts_crop, c_crop = imgutils.crop_norescale(img_origin, ele_copy)
img_out, pts_out, c_out = imgutils.change_resolu(img_crop, ary_pts_crop, c_crop, (256,256))
heatmaps = imgutils.generate_heatmaps(img_out, pts_out, sigma_valu=7)

# img_out = imgutils.random_rotate(img_out)
# heatmaps = imgutils.random_rotate(heatmaps[:,:,:])

# img_out = imgutilsTF.random_flip_tf(img_out)
# heatmaps = imgutilsTF.random_flip_tf(heatmaps[:,:,:])


# Show image and heatmaps
imgutils.show_stack_joints(img_out, pts_out, c_out)
# imgutils.show_heatmaps(img_out, heatmaps, c_out)

# plt.figure(1)
# plt.imshow(img_out)
# plt.show()