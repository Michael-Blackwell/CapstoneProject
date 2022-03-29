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
from tensorflow.keras.layers import Conv2D, Softmax, Permute, BatchNormalization, Activation, Dropout, MaxPooling2D, Conv2DTranspose




def INF(B, H, W):
    vec_1d = tf.repeat(tf.constant(float('inf')), repeats=H)
    diag = -tf.linalg.diag(vec_1d)
    diag = tf.reshape(diag, (1, H, H))
    ddiag = tf.tile(diag, (B * W, 1, 1))
    return ddiag
    # return tf.tile(-tf.reshape(-tf.linalg.diag(tf.repeat(tf.constant(float('inf')), repeats=H)), (1, H, H)), (B * W, 1, 1))


class CrissCrossAttention(tf.keras.Model):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, gamma=0.0):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = Conv2D(filters=in_dim // 8, kernel_size=1, data_format='channels_first')
        self.key_conv = Conv2D(filters=in_dim // 8, kernel_size=1, data_format='channels_first')
        self.value_conv = Conv2D(filters=in_dim, kernel_size=1, data_format='channels_first')
        self.softmax = Softmax(axis=3)
        self.INF = INF
        self.gamma = tf.constant(gamma)

    def call(self, x):
        m_batchsize, _, height, width = x.shape
        proj_query = self.query_conv(x)

        # Prepare Query
        # H
        proj_query_H = Permute((3, 1, 2))(proj_query)
        proj_query_H = tf.reshape(proj_query_H, shape=(m_batchsize * width, -1, height), name='Query_H_Reshape')
        proj_query_H = Permute((2, 1))(proj_query_H)
        # W
        proj_query_W = Permute((2, 1, 3))(proj_query)
        proj_query_W = tf.reshape(proj_query_W, shape=(m_batchsize * height, -1, width), name='Query_W_reshape')
        proj_query_W = Permute((2, 1))(proj_query_W)

        # proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        # proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        # Prepare Key
        # proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_key = self.key_conv(x)
        # H
        proj_key_H = Permute((3, 1, 2))(proj_key)
        proj_key_H = tf.reshape(proj_key_H, (m_batchsize * width, -1, height), name='key_H_reshape')
        # W
        proj_key_W = Permute((2, 1, 3))(proj_key)
        proj_key_W = tf.reshape(proj_key_W, (m_batchsize * height, -1, width), name='key_W_reshape')

        # Prepare Value
        # proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        # H
        proj_value_H = Permute((3, 1, 2))(proj_value)
        proj_value_H = tf.reshape(proj_value_H, (m_batchsize * width, -1, height))
        # W
        proj_value_W = Permute((2, 1, 3))(proj_value)
        proj_value_W = tf.reshape(proj_value_W, (m_batchsize * height, -1, width))

        # Energy
        temp = self.INF(m_batchsize, height, width)
        # torch.bmm(proj_query_H, proj_key_H) + temp
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
        att_W = tf.reshape(concate[:, :, :, height:height + width], (m_batchsize * height, width, width))
        att_W = Permute((2, 1))(att_W)

        # Out
        # H
        # out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_H = tf.keras.layers.Dot(axes=(2, 1))([proj_value_H, att_H])
        out_H = tf.reshape(out_H, (m_batchsize, width, -1, height))
        out_H = Permute((2, 3, 1))(out_H)

        # W
        # out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out_W = tf.keras.layers.Dot(axes=(2, 1))([proj_value_W, att_W])
        out_W = tf.reshape(out_W, (m_batchsize, height, -1, width))
        out_W = Permute((2, 1, 3))(out_W)

        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, filters, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=1, bias=False)
        self.bn1 = BatchNormalization(axis=1)
        self.conv2 = Conv2D(filters, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNormalization(axis=1)
        self.conv3 = Conv2D(filters * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNormalization(axis=1)
        self.relu = ReLU(inplace=False)
        self.relu_inplace = ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle



if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = tf.constant(np.random.rand(2, 64, 5, 8))
    out = model(x)
    print(out.shape)
