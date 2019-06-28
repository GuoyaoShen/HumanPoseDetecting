
from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

# from pose.utils.osutils import *
# from pose.utils.imutils import *
# from pose.utils.transforms import *



# ========================= Check Files in json =========================
with open('mpii_annotations.json') as anno_file:
    anno = json.load(anno_file)

list_train, list_val = [], []
for idx, val in enumerate(anno):
    if val['isValidation'] == True:
        list_val.append(idx)
    else:
        list_train.append(idx)


# View list of anno, 25204 in total
# print('list_val len:', len(list_val))  #2958
# print('list_val:', list_val)
# print('list_train len:', len(list_train))  #22246
# print('list_train:', list_train)



ele = anno[list_val[6]]
# print('ele:', ele)
for key in ele:
    print('KEY', key, np.array(ele[key]))
list_joint = np.array(ele['joint_self'])
print('list_joint.shape:', list_joint.shape)
# print('list_joint:', list_joint)


# Search for specific photo
print('============================================================================')
for ele in anno:
    if ele['img_paths'] == '002047846.jpg':
        print('FOUNDED')
        print('ele:', ele)
        print(np.array(ele['joint_self']).shape)
        print(ele['joint_self'])



# ========================= Check Files in json =========================