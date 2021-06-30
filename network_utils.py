#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import \
    Conv2D, Conv2DTranspose, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from tensorflow_addons.layers import GroupNormalization


# ~~~~~~~~~~ U-Net ~~~~~~~~~~

class U_double_conv(Model):
    def __init__(self, in_ch, out_ch):
        super(U_double_conv, self).__init__()
        self.conv = Sequential(
            layers=(
                Conv2D(out_ch, 3, padding='same'),
                BatchNormalization(),
                ReLU(),
                Conv2D(out_ch, 3, padding='same'),
                BatchNormalization(),
                ReLU()
            )
        )

    def call(self, x):
        x = self.conv(x)
        return x


class inconv(Model):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = U_double_conv(in_ch, out_ch)

    def call(self, x):
        x = self.conv(x)
        return x


class U_down(Model):
    def __init__(self, in_ch, out_ch):
        super(U_down, self).__init__()
        self.mpconv = Sequential(
            layers=(
                MaxPool2d(pool_size=(2, 2), strides=(2, 2)),
                U_double_conv(in_ch, out_ch)
            )
        )

    def call(self, x):
        x = self.mpconv(x)
        return x


class U_up(Model):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(U_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = UpSampling2D(size=(2, 2), mode='bilinear')                    # align_corners missing
        else:
            self.up = Conv2DTranspose(in_ch//2, 2, strides=(2, 2))

        self.conv = U_double_conv(in_ch, out_ch)

    def call(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        paddings = tf.constant([diffY, 0], [diffX, 0])                              # Pay attention to RAM

        x1 = tf.pad(x1, paddings)                                                   # ?
        x = tf.concat([x2, x1], axis=1)

        x = self.conv(x)
        return x


# ~~~~~~~~~~ RU-Net ~~~~~~~~~~

class RU_double_conv(Model):
    def __init__(self, in_ch, out_ch):
        super(RU_double_conv, self).__init__()
        self.conv = Sequential(
            layers=(
                Conv2D(out_ch, 3, padding='same'),
                BatchNormalization(),
                ReLU(),
                Conv2D(out_ch, 3, padding='same'),
                BatchNormalization()
            )
        )

    def call(self, x):
        x = self.conv(x)
        return x


class RU_first_down(Model):
    def __init__(self, in_ch, out_ch):
        super(RU_first_down, self).__init__()
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = ReLU()
        self.res_conv = Sequential(
            layers=(
                Conv2D(out_ch, 1, use_bias=False),
                BatchNormalization()
            )
        )

    def call(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_down(Model):
    def __init__(self, in_ch, out_ch):
        super(RU_down, self).__init__()
        self.maxpool = MaxPool2d(pool_size=(3, 3), strides=(2, 2), padding='same')
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = ReLU()
        self.res_conv = Sequential(
            layers=(
                Conv2D(out_ch, 1, use_bias=False),
                BatchNormalization()
            )
        )

    def call(self, x):
        x = self.maxpool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_up(Model):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RU_up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  UpSampling2D hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = UpSampling2D(size=(2, 2), mode='bilinear')                    # align_corners missing
        else:
            self.up = Conv2DTranspose(in_ch//2, 2, strides=(2, 2))

        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = ReLU()
        self.res_conv = Sequential(
            layers=(
                Conv2D(out_ch, 1, use_bias=False),
                GroupNormalization()
            )
        )

    def call(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        paddings = tf.constant([diffY, 0], [diffX, 0])                              # Pay attention to RAM

        x1 = tf.pad(x1, paddings)                                                   # ?
        x = tf.concat([x2, x1], axis=1)

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


# ~~~~~~~~~~ RRU-Net ~~~~~~~~~~

class RRU_double_conv(Model):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = Sequential(
            layers=(
                    Conv2D(out_ch, 3, padding='same', dilation_rate=(2, 2)),                # padding = 2 ?
                    GroupNormalization(),
                    ReLU(),
                    Conv2D(out_ch, 3, padding='same', dilation_rate=(2, 2)),                # padding = 2 ?
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
        self.pool = MaxPool2d(pool_size=(3, 3), strides=(2, 2), padding='same')

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
            self.up = UpSampling2D(size=(2, 2), mode='bilinear')                    # align_corners missing
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
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        paddings = tf.constant([diffY, 0], [diffX, 0])                              # Pay attention to RAM

        x1 = tf.pad(x1, paddings)                                                   # ?

        x = self.relu(tf.concat([x2, x1], axis=1))

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


# !!!!!!!!!!!! Universal functions !!!!!!!!!!!!

class outconv(Model):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = Conv2D(out_ch, 1)

    def call(self, x):
        x = self.conv(x)
        return x
