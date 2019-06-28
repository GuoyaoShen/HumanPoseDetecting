import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import imgutils
import os

from utils import imgutils
from utils import imgutilsTF
from utils import Dataset_Make_Utils as dsmutils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


num_subset = 4
name_file = 'datasets/np__train_imgs_'+str(num_subset)+'.npy'
train_imgs = np.load(name_file)
img = train_imgs[999, ...]
plt.figure(1)
plt.imshow(img)
plt.show()