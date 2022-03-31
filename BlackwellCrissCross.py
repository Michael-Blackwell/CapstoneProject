"""
Author: mike
Created: 3/30/22

"""
"""
Author: mike
Created: 3/27/22

"""

import tensorflow_addons as tfa
import tensorboard
import pandas as pd
from ModelFunctions import *
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, MultiHeadAttention, ReLU
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Softmax, Permute, BatchNormalization, Activation, Dropout, MaxPooling2D, \
    Conv2DTranspose


def INF(B, H, W):
    vec_1d = tf.repeat(tf.constant(float('inf')), repeats=H)
    diag = -tf.linalg.diag(vec_1d)
    diag = tf.reshape(diag, (1, H, H))
    ddiag = tf.tile(diag, (B * W, 1, 1))
    return ddiag
    # return tf.tile(-tf.reshape(-tf.linalg.diag(tf.repeat(tf.constant(float('inf')), repeats=H)), (1, H, H)), (B * W, 1, 1))


class CrissCrossAttention(tf.keras.layers.Layer):
    """ Criss-Cross Attention Module"""

    def __init__(self, image_size, gamma=0.05):
        super(CrissCrossAttention, self).__init__()

        # Want to reduce dimensions to 1/8 of input image dimensions
        self.query_conv = Conv2D(filters=image_size // 8, kernel_size=1)
        self.key_conv = Conv2D(filters=image_size // 8, kernel_size=1)
        self.value_conv = Conv2D(filters=image_size, kernel_size=1)
        self.softmax = Softmax(axis=3)
        self.INF = INF
        self.gamma = tf.constant(gamma)

    def call(self, x):
        # Permute(1,3,2) is a transposition
        m_batchsize, height, width, channels = x.shape

        # Obtain Query, Key, and Value feature maps
        Query = self.query_conv(x)
        Key = self.key_conv(x)
        Value = self.value_conv(x)

        # Prepare Query
        proj_query_H = Permute((2, 3, 1))(Query)
        proj_query_H = tf.reshape(proj_query_H, (m_batchsize * width, -1, height))
        proj_query_H = Permute((2, 1))(proj_query_H)

        proj_query_W = Permute((1, 3, 2))(Query)
        proj_query_W = tf.reshape(proj_query_W, (m_batchsize * height, -1, width))
        proj_query_W = Permute((2, 1))(proj_query_W)

        # Prepare Key
        proj_key_H = Permute((2, 3, 1))(Key)
        proj_key_H = tf.reshape(proj_key_H, (m_batchsize * width, -1, height))

        proj_key_W = Permute((1, 3, 2))(Key)
        proj_key_W = tf.reshape(proj_key_W, (m_batchsize * height, -1, width))

        # Prepare Value
        proj_value_H = Permute((2, 3, 1))(Value)
        proj_value_H = tf.reshape(proj_value_H, (m_batchsize * width, -1, height))

        proj_value_W = Permute((1, 3, 2))(Value)
        proj_value_W = tf.reshape(proj_value_W, (m_batchsize * height, -1, width))

        # Apply Affinity Operation
        temp = self.INF(m_batchsize, height, width)
        # H
        energy_H = tf.keras.layers.Dot(axes=(2, 1))([proj_query_H, proj_key_H]) + temp
        energy_H = tf.reshape(energy_H, (m_batchsize, width, height, height))
        energy_H = Permute((2, 1, 3))(energy_H)
        # W
        energy_W = tf.keras.layers.Dot(axes=(2, 1))([proj_query_W, proj_key_W])
        energy_W = tf.reshape(energy_W, (m_batchsize, height, width, width))

        concate = self.softmax(tf.concat([energy_H, energy_W], 3))


        # Attention
        # H
        att_H = Permute((2, 1, 3))(concate[:, :, :, 0:height])
        att_H = tf.reshape(att_H, (m_batchsize * width, height, height))
        att_H = Permute((2, 1))(att_H)

        # W
        att_W = concate[:, :, :, height:height + width]
        att_W = tf.reshape(att_W, (m_batchsize * height, width, width))
        att_W = Permute((2, 1))(att_W)

        # Out
        # H
        out_H = tf.keras.layers.Dot(axes=(2, 1))([proj_value_H, att_H])
        out_H = tf.reshape(out_H, (m_batchsize, width, -1, height))
        out_H = Permute((3, 1, 2))(out_H)
        # W
        out_W = tf.keras.layers.Dot(axes=(2, 1))([proj_value_W, att_W])
        out_W = tf.reshape(out_W, (m_batchsize, height, -1, width))
        out_W = Permute((1, 3, 2))(out_W)


        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class RCCAModule(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 3
        self.conva = tf.keras.Sequential(layers=[Conv2D(filters=inter_channels, kernel_size=3, padding='same',
                                                        use_bias=False, data_format='channels_first'),  # TODO padding
                                                 BatchNormalization(axis=1)])  # TODO normalize by features axis
        self.cca = CrissCrossAttention(inter_channels)  # todo should be h/w?
        self.convb = tf.keras.Sequential(layers=[Conv2D(filters=inter_channels, kernel_size=3, padding='same',
                                                        use_bias=False, data_format='channels_first'),  # TODO padding
                                                 BatchNormalization(axis=1)])  # TODO normalize by features axis

        self.bottleneck = tf.keras.Sequential(layers=[
            Conv2D(filters=out_channels, kernel_size=3, padding='same', use_bias=False, data_format='channels_first'),
            # TODO padding
            BatchNormalization(axis=1),  # TODO normalize by features axis
            Dropout(0.1),
            Conv2D(filters=num_classes, kernel_size=1, use_bias=True, data_format='channels_first')]
        )

    def call(self, x, recurrence=1):
        # Convert from (H,W,C) to (C,H,W)
        # x = Permute((3, 1, 2))(x)
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(tf.concat([x, output], 1))
        # # Convert back to (H,W,C)
        # output = Permute((2, 3, 1))(output)
        return output


if __name__ == '__main__':
    in_channels = 16
    model = CrissCrossAttention(in_channels)
    x = tf.constant(np.random.rand(2, 100, 110, in_channels))
    out = model(x)
    print(out.shape)