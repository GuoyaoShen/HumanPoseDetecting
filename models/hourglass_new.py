import tensorflow as tf
from tensorflow import keras as K

# Channel first: (N,C,H,W)
# state_channel, state_chaxis = 'channels_first', 1  # (N,C,H,W)
state_channel, state_chaxis = 'channels_last', 3  # (N,H,W,C)

class Residual(K.Model):
    # CHANNELS: 2*planes_mid -> 2*planes_mid
    # This Residual net does not change the # of channels of the input, always be 2*planes_mid
    # The input # of channels 'planes_in' must be equal to '2*planes_mid', the # of output channels is 2*planes_mid
    expansion = 2
    def __init__(self, planes_mid, stride=1, downsample=None):
        super(Residual, self).__init__()
        self.bn1 = K.layers.BatchNormalization(axis=state_chaxis)
        self.conv1 = K.layers.Conv2D(filters=planes_mid,kernel_size=1,
                                     data_format=state_channel,use_bias=True)  # planes_in -> planes_mid
        self.bn2 = K.layers.BatchNormalization(axis=state_chaxis)
        self.conv2 = K.layers.Conv2D(filters=planes_mid, kernel_size=3, strides=stride,
                                     padding='same',data_format=state_channel,use_bias=True)  # planes_mid -> planes_mid
        self.bn3 = K.layers.BatchNormalization(axis=state_chaxis)
        self.conv3 = K.layers.Conv2D(filters=self.expansion*planes_mid,kernel_size=1,
                                     data_format=state_channel,use_bias=True)  # planes_in -> planes_mid
        self.relu = K.layers.ReLU()
        self.downsample = downsample

    def call(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class HourGlass(K.Model):
    # block: input model block (class) to use. (Residual class in this model)
    # block_num: define the # of stacked blocks, to be a independent sub-net
    # planes_mid: define the mid planes to input to blocks class
    # depth: define the depth of the HourGlass sub-net
    def __init__(self, block, block_num, planes_mid, depth):
        super(HourGlass, self).__init__()
        self.block = block
        self.depth = depth

        self.maxpool2d = K.layers.MaxPool2D(pool_size=2, strides=2, padding='same', data_format=state_channel)
        self.upsampling = K.layers.UpSampling2D(size=2, data_format=state_channel)
        self.hourglass = self._make_hourglass(block, block_num, planes_mid, depth)

    def _residual_stack(self, block, block_num, planes_mid):
        # Return a stacked net of block(Residual) with the num of block_num
        list_res = []
        for i in range(block_num):
            list_res.append(block(planes_mid))
        return K.Sequential(list_res)

    def _make_hourglass(self, block, block_num, planes_mid, depth):
        # Function to make a hourglass net with the depth of 'depth'
        list_hourglass = []
        for i in range(depth):
            list_res_stack = []
            for j in range(3):
                list_res_stack.append(self._residual_stack(block, block_num, planes_mid))
            if i == 0:
                list_res_stack.append(self._residual_stack(block, block_num, planes_mid))
            list_hourglass.append(list_res_stack)
        return list_hourglass

    def _hourglass_forward(self, x, n):
        up1 = self.hourglass[n - 1][0](x)

        low1 = self.maxpool2d(x)
        low1 = self.hourglass[n - 1][1](low1)
        if n > 1:
            low2 = self._hourglass_forward(low1, n-1)
        else:
            low2 = self.hourglass[n - 1][3](low1)
        low3 = self.hourglass[n - 1][2](low2)
        up2 = self.upsampling(low3)

        out = up1 + up2
        return out

    def call(self, x):
        return self._hourglass_forward(x, self.depth)

class HourGlassNet(K.Model):
    def __init__(self, block, block_num, stack_num, class_num):
        super(HourGlassNet, self).__init__()
        self.planes_in = 64
        self.feats = 128
        # self.block_num = block_num
        self.stack_num = stack_num

        self.conv1 = K.layers.Conv2D(filters=self.planes_in, kernel_size=7, strides=2,
                                     padding='same', data_format=state_channel, use_bias=True)
        self.bn1 = K.layers.BatchNormalization(axis=state_chaxis)
        self.relu = K.layers.ReLU()

        self.layer1 = self._residual_stack(block, 1, self.planes_in)  # inplanes -> 2*inplanes
        self.layer2 = self._residual_stack(block, 1, self.planes_in)  # 2*inplanes -> 4*inplanes
        self.layer3 = self._residual_stack(block, 1, self.feats)  # 4*inplanes -> 2*num_feats
        self.maxpool = K.layers.MaxPool2D(pool_size=2, strides=2,
                                          padding='same', data_format=state_channel)  # (C,H,W)->(C,H/2,W/2)

        # Make hourglass model
        ch = self.feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(stack_num):
            hg.append(HourGlass(block, block_num, self.feats, 4))  # 2*num_feats -> 2*num_feats
            res.append(self._residual_stack(block, block_num, self.feats))  # 4*inplanes -> 2*num_feats ()
            fc.append(self._make_fc(ch))  # 2*num_feats -> 2*num_feats
            score.append(K.layers.Conv2D(class_num, kernel_size=1, padding='same',
                                         data_format=state_channel, use_bias=True))
            if i < stack_num-1:  # When not the final stack
                fc_.append(K.layers.Conv2D(ch, kernel_size=1, padding='same',
                                           data_format=state_channel, use_bias=True))
                score_.append(K.layers.Conv2D(ch, kernel_size=1, padding='same',
                                              data_format=state_channel, use_bias=True))
        self.hg = hg
        self.res = res
        self.fc = fc
        self.score = score
        self.fc_ = fc_
        self.score_ = score_


    def _residual_stack(self, block, block_num, planes_mid, stride=1):
        downsample = None
        if stride != 1 or self.planes_in != block.expansion*planes_mid:
            downsample = K.layers.Conv2D(filters=block.expansion*planes_mid, kernel_size=1, strides=stride,
                                         padding='same', data_format=state_channel, use_bias=True)
        list_res_stack = []
        list_res_stack.append(block(planes_mid, stride=stride, downsample=downsample))
        self.planes_in = block.expansion*planes_mid
        for i in range(1, block_num):
            list_res_stack.append(block(planes_mid))
        return K.Sequential(list_res_stack)

    def _make_fc(self, planes_out):
        bn = K.layers.BatchNormalization(axis=state_chaxis)
        conv = K.layers.Conv2D(planes_out, kernel_size=1, padding='same', data_format=state_channel, use_bias=True)
        return K.Sequential([
            bn,
            conv,
            self.relu
        ])

    def call(self, x):
        out = []
        x = self.conv1(x)  # 3 -> self.inplanes, to (H/2,W/2)
        x = self.bn1(x)  # self.inplanes -> self.inplanes
        x = self.relu(x)  # self.inplanes -> self.inplanes

        x = self.layer1(x)  # self.inplanes -> 2*self.inplanes
        x = self.maxpool(x)  # to (H/4,W/4), 2*self.inplanes -> 2*self.inplanes
        x = self.layer2(x)  # 2*inplanes -> 4*inplanes
        x = self.layer3(x)  # 4*inplanes -> 2*num_feats

        for i in range(self.stack_num):
            y = self.hg[i](x)  # 2*num_feats -> 2*num_feats
            y = self.res[i](y)  # 4*inplanes -> 2*num_feats (256 -> 256)
            y = self.fc[i](y)  # 2*num_feats -> 2*num_feats
            score = self.score[i](y)  # 2*num_feats -> num_class
            out.append(score)
            # out = score
            if i < self.stack_num - 1:
                fc_ = self.fc_[i](y)  # 2*num_feats -> 2*num_feats
                score_ = self.score_[i](score)  # num_class ->2*num_feats
                x = x + fc_ + score_
        return out

def hg(num_stack=1, num_block=1, num_class=10):
    # print(range(num_stack))
    model = HourGlassNet(Residual, block_num=num_block, stack_num=num_stack, class_num=num_class)
    return model

