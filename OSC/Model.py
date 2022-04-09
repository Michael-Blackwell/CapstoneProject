"""
Author: mike
Created: 3/31/22

"""

from Functions import *
import tensorflow_addons as tfa
import tensorboard
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Permute, Softmax, Conv2DTranspose, MaxPool2D, \
    Activation, Concatenate
from tensorflow.keras.metrics import SpecificityAtSensitivity, Recall, AUC
import logging


def INF(B, H, W):
    vec_1d = tf.repeat(tf.constant(float('inf')), repeats=H)
    diag = -tf.linalg.diag(vec_1d)
    diag = tf.reshape(diag, (1, H, H))
    ddiag = tf.tile(diag, (B * W, 1, 1))
    return ddiag


class CrissCrossAttention(tf.keras.layers.Layer):
    """Criss-Cross Attention Module"""

    def __init__(self):
        super(CrissCrossAttention, self).__init__()
        self.softmax = Softmax(axis=3)
        self.INF = INF

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        self.query_conv = Conv2D(filters=input_shape[-1] // 8, kernel_size=1)
        self.key_conv = Conv2D(filters=input_shape[-1] // 8, kernel_size=1)
        self.value_conv = Conv2D(filters=input_shape[-1], kernel_size=1)
        self.gamma = tf.Variable(tf.constant(0.05))

    def call(self, x_in):
        """

        :param x_in:
        :return:
        """
        m_batchsize, height, width, channels = x_in.shape

        # Obtain Query, Key, and Value feature maps
        Query = self.query_conv(x_in)
        Key = self.key_conv(x_in)
        Value = self.value_conv(x_in)

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
        temp = INF(m_batchsize, height, width)
        # H
        energy_H = tf.matmul(proj_query_H, proj_key_H) + temp
        # energy_H = tf.keras.layers.Dot(axes=(2, 1))([proj_query_H, proj_key_H]) + temp
        energy_H = tf.reshape(energy_H, (m_batchsize, width, height, height))
        energy_H = Permute((2, 1, 3))(energy_H)
        # W
        energy_W = tf.matmul(proj_query_W, proj_key_W)
        # energy_W = tf.keras.layers.Dot(axes=(2, 1))([proj_query_W, proj_key_W])
        energy_W = tf.reshape(energy_W, (m_batchsize, height, width, width))

        concate = Softmax(axis=3)(tf.concat([energy_H, energy_W], 3))

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
        out_H = tf.matmul(proj_value_H, att_H)
        # out_H = tf.keras.layers.Dot(axes=(2, 1))([proj_value_H, att_H])
        out_H = tf.reshape(out_H, (m_batchsize, width, -1, height))
        out_H = Permute((3, 1, 2))(out_H)
        # W
        out_W = tf.matmul(proj_value_W, att_W)
        # out_W = tf.keras.layers.Dot(axes=(2, 1))([proj_value_W, att_W])
        out_W = tf.reshape(out_W, (m_batchsize, height, -1, width))
        out_W = Permute((1, 3, 2))(out_W)

        return self.gamma * (out_H + out_W) + x_in


def build_model(image_size, filters=32, kernelsize=3, batch_size=2, recurrence=2, name='CCModel'):
    num_classes = 2

    # Convolution parameters
    conv_params = dict(kernel_size=(3, 3), activation="relu",
                       padding="same",
                       kernel_initializer="he_uniform")

    # Transposed convolution parameters
    deconv_params = dict(kernel_size=(2, 2), strides=(2, 2),
                         padding="same")

    # Define the model
    image = tf.keras.Input(shape=image_size, name='image', batch_size=batch_size)

    # Convolution 1 (510x510)
    x = Conv2D(filters=filters, name='conv_1_1', **conv_params)(image)
    x1 = Conv2D(filters=filters, name='conv_1_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x1)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Convolution 2 (253x253)
    x = Conv2D(filters=filters * 2, name='conv_2_1', **conv_params)(x)
    x2 = Conv2D(filters=filters * 2, name='conv_2_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x2)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Convolution 3 (124x124)
    x = Conv2D(filters=filters * 4, name='conv_3_1', **conv_params)(x)
    x3 = Conv2D(filters=filters * 4, name='conv_3_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x3)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Convolution 4 (60x60)
    x = Conv2D(filters=filters * 8, name='conv_4_1', **conv_params)(x)
    x4 = Conv2D(filters=filters * 8, name='conv_4_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x4)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Convolution 5
    x = Conv2D(filters=filters * 16, name='conv_5_1', **conv_params)(x)
    x5 = Conv2D(filters=filters * 16, name='conv_5_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x5)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Deconvolution 1
    x = Conv2DTranspose(filters=filters * 8, name='deconv_1_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x5])

    x = Conv2D(filters=filters * 8, name='deconv_1_2', **conv_params)(x)
    x = Conv2D(filters=filters * 8, name='deconv_1_3', **conv_params)(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Deconvolution 2
    x = Conv2DTranspose(filters=filters * 4, name='deconv_2_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x4])

    x = Conv2D(filters=filters * 4, name='deconv_2_2', **conv_params)(x)
    x = Conv2D(filters=filters * 4, name='deconv_2_3', **conv_params)(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)
    # x = Dropout(0.1)(x)

    # RCCA
    x_attn = Conv2D(filters=filters * 4, name='ConvA', **conv_params)(x)  # TODO padding
    x_attn = BatchNormalization()(x_attn)

    # Criss Cross Attention
    for i in range(recurrence):
        x_attn = CrissCrossAttention()(x_attn)

    x_attn = Conv2D(filters=filters * 2, kernel_size=1, padding='same', use_bias=False)(
        tf.concat([x, x_attn], 3))  # TODO padding
    x_attn = BatchNormalization(axis=3)(x_attn)
    x_attn = Dropout(0.1)(x_attn)
    # x_attn = Conv2D(filters=num_classes, kernel_size=1, use_bias=True)(x_attn)

    # Deconvolution 3
    x = Conv2DTranspose(filters=filters * 2, name='deconv_3', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x3])

    x = Conv2D(filters=filters * 2, name='deconv_3_2', **conv_params)(x)
    x = Conv2D(filters=filters * 2, name='deconv_3_3', **conv_params)(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.1)(x)

    # Deconvolution 4
    x = Conv2DTranspose(filters=filters, name='deconv_4_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x2])

    x = Conv2D(filters=filters, name='deconv_4_2', **conv_params)(x)
    x = Conv2D(filters=filters, name='deconv_4_3', **conv_params)(x)
    x = BatchNormalization(axis=3)(x)

    # Deconvolution 5
    x = Conv2DTranspose(filters=filters, name='deconv_5_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x1])

    x = Conv2D(filters=filters, name='deconv_5_2', **conv_params)(x)
    x = Conv2D(filters=num_classes, name='deconv_5_3', **conv_params)(x)
    x = BatchNormalization(axis=3)(x)

    # x = Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(x)
    mask = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='mask')(x)

    # Define the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask],  # melanoma, keratosis],
                           name=name)

    # Compile model
    model.compile(
        loss=jaccard_loss,
        # tfa.losses.sigmoid_focal_crossentropy,  # 'categorical_crossentropy',  # tfa.losses.sigmoid_focal_crossentropy,
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=[dice_coef,
                 jaccard_distance,
                 myIOU(),
                 SpecificityAtSensitivity(0.5),
                 Recall(),
                 AUC()
                 ]  # , tfa.metrics.MultiLabelConfusionMatrix],
    )

    return model


def callbacks(ts: pd.Timestamp) -> tuple:
    """
    Define callbacks and store them in a timestamped folder.

    :return:
    """

    # Create folders.
    tb_path = Path(f'/media/storage/Capstone1/Callbacks/tensorboard/{ts}')
    ckpt_path = Path(f'/media/storage/Capstone1/Callbacks/checkpoints/{ts}')
    early_stop_path = Path(f'/media/storage/Capstone1/Callbacks/earlystopping/{ts}')

    ckpt_path.mkdir()
    tb_path.mkdir()
    early_stop_path.mkdir()

    # Define callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path))
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # TODO add confusion matrix & jaccard/dice callbacks

    call_backs = [tensorboard_cb, checkpoint_cb, early_stopping_cb]

    return call_backs, tb_path


def train_model(model, data_pipe, epochs: int, ts):
    logger = logging.getLogger(__name__)
    calls, tb_path = callbacks(ts)
    logger.log(logging.INFO, 'Callbacks Created')

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    # tb = tensorboard.program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', str(tb_path)])
    # url = tb.launch()

    # Change to 'CPU:0' to use CPU instead of GPU
    with tf.device('GPU:0'):
        model.fit(
            data_pipe.train,
            validation_data=data_pipe.val,
            epochs=epochs,
            callbacks=calls
        )

    logger.log(logging.INFO, 'Training Complete')

    return model

