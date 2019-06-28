import torch
import torch.nn as nn

a=torch.rand(2,4,3)
print('a.SHAPE', a.shape)
print('a', a)
b = a.split(1, 1)
# print('b.SHAPE', b.shape)
print('b', b)