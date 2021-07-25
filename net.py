# Implementation in Tensorflow-2. of "https://github.com/yelusaleng/RRU-Net"

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import \
    Conv2D, Conv2DTranspose, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from tensorflow_addons.layers import GroupNormalization


class RRU_Net(Model):
    def __init__(self, n_channels=3, n_classes=1):
        super(RRU_Net, self).__init__()
        self.down = RRU_first_down(n_channels, 32)
        self.down1 = RRU_down(32, 64)
        self.down2 = RRU_down(64, 128)
        self.down3 = RRU_down(128, 256)
        self.down4 = RRU_down(256, 256)
        self.up1 = RRU_up(512, 128)
        self.up2 = RRU_up(256, 64)
        self.up3 = RRU_up(128, 32)
        self.up4 = RRU_up(64, 32)
        self.out = outconv(32, n_classes)

    def call(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class RRU_double_conv(Model):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = Sequential(
            layers=(
                    Conv2D(out_ch, 3, padding='same', dilation_rate=(2, 2)),               
                    GroupNormalization(),
                    ReLU(),
                    Conv2D(out_ch, 3, padding='same', dilation_rate=(2, 2)),                
                    GroupNormalization()
            )
        )

    def call(self, x):
        x = self.conv(x)
        return x


class RRU_first_down(Model):
    def __init__(self, in_ch, out_ch):
        super(RRU_first_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = ReLU()

        self.res_conv = Sequential(
            layers=(
                Conv2D(out_ch, 1, use_bias=False),
                GroupNormalization()
            )
        )
        self.res_conv_back = Sequential(
            layers=(
                Conv2D(in_ch, 1, use_bias=False)             
            )
        )

    def call(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = tf.math.multiply(1 + tf.math.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_down(Model):
    def __init__(self, in_ch, out_ch):
        super(RRU_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        self.res_conv = Sequential(
            layers=(
                Conv2D(out_ch, 1, use_bias=False),
                GroupNormalization()
            )
        )
        self.res_conv_back = Sequential(
            layers=(
                Conv2D(in_ch, 1, use_bias=False)
            )
        )

    def call(self, x):
        x = self.pool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = tf.math.multiply(1 + tf.math.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up(Model):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RRU_up, self).__init__()
        if bilinear:
            self.up = UpSampling2D(size=(2, 2), mode='bilinear')                    
        else:
            self.up = Sequential(
                layers=(
                    Conv2DTranspose(in_ch//2, 2, strides=(2, 2)),
                    GroupNormalization()
                )
            )

        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = ReLU()

        self.res_conv = Sequential(
            layers=(
                Conv2D(out_ch, 1, use_bias=False),
                GroupNormalization()
            )
        )
        self.res_conv_back = Sequential(
            layers=(
                Conv2D(in_ch, 1, use_bias=False)
            )
        )

    def call(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.shape[2] - x1.shape[2]
        diffY = x2.shape[3] - x1.shape[3]
        paddings = [[0, 0], [0, 0], [diffX, 0], [diffY, 0]]

        x1 = tf.pad(x1, paddings)

        x = self.relu(tf.concat([x2, x1], axis=-1))

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = tf.math.multiply(1 + tf.math.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class outconv(Model):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = Conv2D(out_ch, 1)

    def call(self, x):
        x = self.conv(x)
        return x
