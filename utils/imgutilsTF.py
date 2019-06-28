import tensorflow as tf
import tensorflow.image as tfimage

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def random_rotate_tf(img, out_np=True):
    # -img: np array or Tensor, (H,W,C) or (N,H,W,C)
    # -out_np: whether output is np array or Tensor
    if tf.is_tensor(img) == False:
        img = tf.convert_to_tensor(img)

    pass

def random_flip_tf(img, out_np=True):
    # -img: np array or Tensor, (H,W,C) or (N,H,W,C)
    # -out_np: whether output is np array or Tensor
    if tf.is_tensor(img) == False:
        img = tf.convert_to_tensor(img)
    img = tfimage.random_flip_left_right(img)
    img = tfimage.random_flip_up_down(img)
    if out_np == True:
        img = img.numpy()
    return img