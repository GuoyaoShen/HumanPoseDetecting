import tensorflow as tf
import tensorflow.keras as K
from models.hourglass_new import Residual, HourGlass, HourGlassNet, hg

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print(tf.__version__)


# ========================= Test for Residual class =========================
# x = tf.random.normal([12,6,128,128])
# net_residual = Residual(3)
#
# out = net_residual(x)
# print(out.shape)
# net_residual.summary()
# ========================= Test for Residual class =========================



# ========================= Test for HourGlass class =========================
# ==== Sequential List Test ====
# seq = [
#     # K.layers.Conv2D(filters=16,kernel_size=1,data_format='channels_first',use_bias=True),
#     K.layers.Dense(32, input_shape=(10,)),
#     K.layers.Dense(10)
# ]
# net_seq1 = K.Sequential(seq)
# net_seq2 = K.Sequential(seq)
# net_seq = K.Sequential([net_seq1, net_seq2])
#
#
# # x = tf.random.normal([12,6,128,128])
# x = tf.random.normal([12, 10])
# # net_test = seq[0]
#
# out = net_seq(x)
# # net_seq.build()
# net_seq.summary()
# ==== Sequential List Test ====


#============================
# net_hourglass = HourGlass(Residual, block_num=2, planes_mid=3, depth=4)
# x = tf.random.normal([12,6,128,128])
#
# out = net_hourglass(x)
# net_hourglass.summary()
# ========================= Test for HourGlass class =========================






# ========================= Test for HourGlassNet class =========================
# x = tf.random.normal([12,3,256,256])
# x = tf.random.normal([32,256,256,3])
x = tf.random.normal([12,256,256,3])
net_hg = hg(num_stack=8, num_block=1, num_class=16)
# net_hg = hg()
# [out] = net_hg(x)
out = net_hg(x)
out = out[-1]

# out = tf.squeeze(out, axis=0)
# print('out', out)
print('out.SHAPE', out.shape)
net_hg.summary()
# ========================= Test for HourGlassNet class =========================





# ========================= Test for Dataset class =========================
# (x, y), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
# x = tf.convert_to_tensor(x)
# y = tf.convert_to_tensor(y)
# x_test = tf.convert_to_tensor(x_test)
# y_test = tf.convert_to_tensor(y_test)
# print(x.shape, y.shape)
# print('x:', x)
# print(tf.is_tensor(x))
#
# db = tf.data.Dataset.from_tensor_slices((x, y))
# print(db)
# ========================= Test for Dataset class =========================