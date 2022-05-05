"""

"""
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Permute, Conv2DTranspose, MaxPool2D, \
    Concatenate, Attention, SpatialDropout2D



class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, conv_params: dict, pool_size=2, bnorm_axis=3):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2D(filters=filters, **conv_params)
        self.conv2 = Conv2D(filters=filters, **conv_params)
        self.pool = MaxPool2D(pool_size=pool_size)
        self.bnorm = BatchNormalization(axis=bnorm_axis)

    def call(self, x_in):
        x_in = self.conv1(x_in)
        x_in = self.conv2(x_in)
        x_in = self.pool(x_in)
        x_in = self.bnorm(x_in)

        return x_in

    def get_config(self):
        base_config = super().get_config()
        return base_config


class DeconvBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, deconv_params: dict, conv_params: dict, pool_size=2, bnorm_axis=3):
        super(DeconvBlock, self).__init__()
        self.deconv1 = Conv2DTranspose(filters=filters, **deconv_params)
        self.conv1 = Conv2DTranspose(filters=filters, **conv_params)
        self.conv2 = Conv2DTranspose(filters=filters, **conv_params)
        self.bnorm = BatchNormalization(axis=bnorm_axis)

    def call(self, x_in, skip_con=None):
        x_in = self.deconv1(x_in)
        if skip_con is not None:
            x_in = Concatenate(axis=-1)([x_in, skip_con])
        x_in = self.conv1(x_in)
        x_in = self.conv2(x_in)
        x_in = self.bnorm(x_in)

        return x_in

    def get_config(self):
        base_config = super().get_config()
        return base_config


class CCAttn(tf.keras.layers.Layer):

    def __init__(self, conv_params: dict, use_scale=True):
        super(CCAttn, self).__init__()
        self.use_scale = use_scale
        self.conv_params = conv_params

        self.attn2 = Attention(use_scale=self.use_scale)
        self.gamma = tf.Variable(tf.constant(0.05))
        self.batchnorm = BatchNormalization(axis=3)

    def build(self, input_shape):
        super().build(input_shape)
        self.conv = Conv2D(filters=input_shape[-1], **self.conv_params)

    def call(self, x_in):
        # Row
        x_1 = Permute((3, 1, 2))(x_in)

        # Column
        x_2 = Permute((3, 2, 1))(x_in)

        x_3 = self.attn2([x_2, x_1])
        x_3 = Permute((2, 3, 1))(x_3)

        # weigh attention, concat with input, and normalize
        out = self.conv(tf.concat([(self.gamma * x_3), x_in], axis=3))
        out = self.batchnorm(out)

        return out

    def get_config(self):
        base_config = super().get_config()
        return base_config


class AttentionUNet:

    def __init__(self, image_size: tuple, name, filters=16, drop_rate=0.01):
        # Parameters
        self.name = name
        self.conv_params = dict(kernel_size=(3, 3), activation="relu",
                                padding="same",
                                kernel_initializer="he_uniform")

        self.deconv_params = dict(kernel_size=(2, 2), strides=(2, 2),
                                  padding="same")

        # Input
        self.input1 = tf.keras.Input(shape=image_size, name='image')

        # Convolution Blocks
        self.conv1 = ConvBlock(filters=filters, conv_params=self.conv_params)
        self.conv2 = ConvBlock(filters=filters * 2, conv_params=self.conv_params)
        self.conv3 = ConvBlock(filters=filters * 4, conv_params=self.conv_params)
        self.conv4 = ConvBlock(filters=filters * 8, conv_params=self.conv_params)
        self.conv5 = ConvBlock(filters=filters * 16, conv_params=self.conv_params)

        # Transposed Convolution Blocks
        self.deconv1 = DeconvBlock(filters=filters * 8, conv_params=self.conv_params, deconv_params=self.deconv_params)
        self.deconv2 = DeconvBlock(filters=filters * 4, conv_params=self.conv_params, deconv_params=self.deconv_params)
        self.deconv3 = DeconvBlock(filters=filters * 2, conv_params=self.conv_params, deconv_params=self.deconv_params)
        self.deconv4 = DeconvBlock(filters=filters, conv_params=self.conv_params, deconv_params=self.deconv_params)
        self.deconv5 = DeconvBlock(filters=filters, conv_params=self.conv_params, deconv_params=self.deconv_params)

        # Attention Layers
        self.attn1 = CCAttn(conv_params=self.conv_params)
        self.attn2 = CCAttn(conv_params=self.conv_params)

        # Dropout Layers
        self.drop1 = SpatialDropout2D(drop_rate, data_format='channels_last')
        self.drop2 = SpatialDropout2D(drop_rate, data_format='channels_last')
        self.drop3 = SpatialDropout2D(drop_rate, data_format='channels_last')
        self.drop4 = SpatialDropout2D(drop_rate, data_format='channels_last')
        self.drop5 = SpatialDropout2D(drop_rate, data_format='channels_last')

        # Output Layer
        self.out = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='mask')

    def build_model(self):
        image = self.input1
        x = self.conv1(image)
        x1 = self.drop1(x)

        x = self.conv2(x1)
        x2 = self.drop2(x)

        x = self.conv3(x2)
        x3 = self.drop3(x)

        x = self.conv4(x3)
        x4 = self.drop4(x)

        x = self.conv5(x4)
        x = self.drop5(x)

        # Attention Weighting
        x = self.attn1(x)
        x = self.attn2(x)

        # DeConvolution
        x = self.deconv1(x, x4)
        x = self.deconv2(x, x3)
        x = self.deconv3(x, x2)
        x = self.deconv4(x, x1)
        x = self.deconv5(x)
        mask = self.out(x)

        return tf.keras.Model(inputs=[image], outputs=[mask], name=self.name)
