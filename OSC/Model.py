"""
Author: mike
Created: 3/31/22

"""

from Functions import *
import tensorboard
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Permute, Softmax, Conv2DTranspose, MaxPool2D, \
    Concatenate, Attention, Flatten, Dense
from tensorflow.keras.metrics import Recall, Precision, AUC
import logging


def binary_focal_crossentropy1(
    target,
    output,
    gamma=2.0,
    from_logits=False,
):
  sigmoidal = tf.__internal__.smart_cond.smart_cond(
      from_logits,
      lambda: tf.sigmoid(output),
      lambda: output,
  )
  p_t = (target * sigmoidal) + ((1 - target) * (1 - sigmoidal))
  # Calculate focal factor
  focal_factor = tf.pow(1.0 - p_t, gamma)
  # Binary crossentropy
  bce = tf.keras.backend.binary_crossentropy(
      target=target,
      output=output,
      from_logits=from_logits,
  )
  return focal_factor * bce


def binary_focal_crossentropy(
    y_true,
    y_pred,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.,
    axis=-1,
):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

    def _smooth_labels():
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    y_true = tf.__internal__.smart_cond.smart_cond(label_smoothing,
                                                 _smooth_labels, lambda: y_true)

    return tf.keras.backend.mean(
      binary_focal_crossentropy1(
          target=y_true,
          output=y_pred,
          gamma=gamma,
          from_logits=from_logits,
      ),
      axis=axis,
    )



class SelfAttn(tf.keras.layers.Layer):
    def __init__(self, use_scale=True):
        super(SelfAttn, self).__init__()
        self.use_scale = use_scale

    def build(self, input_shape):
        # self.attn1 = Attention(use_scale=self.use_scale)
        self.attn2 = Attention(use_scale=self.use_scale)
        # self.attn3 = Attention(use_scale=self.use_scale)
        # self.conv = Conv2D(filters=input_shape[-1], kernel_size=1, activation='relu')
        # self.in_shape = input_shape
        self.gamma = tf.Variable(tf.constant(0.05))
        # self.beta = tf.Variable(tf.constant(0.05))

    def call(self, x_in):
        # Row Attention
        x_1 = Permute((3, 1, 2))(x_in)
        # x_1 = self.attn1([x_1, x_1])
        # x_1 = Permute((2, 3, 1))(x_1)

        # Column Attention
        x_2 = Permute((3, 2, 1))(x_in)
        x_3 = self.attn2([x_2, x_1])
        x_3 = Permute((3, 2, 1))(x_3)

        # Row/Col Attention
        # x_3 = self.attn3([x_1, x_2])
        # x_3 = Permute((3, 2, 1))(x_3)

        return tf.concat([(self.gamma * x_3), x_in], axis=3)

    def get_config(self):
        base_config = super().get_config()
        return base_config


def build_model(image_size, filters=32, name='CCModel'):
    num_classes = 2

    # Convolution parameters
    conv_params = dict(kernel_size=(3, 3), activation="relu",
                       padding="same",
                       kernel_initializer="he_uniform")

    # Transposed convolution parameters
    deconv_params = dict(kernel_size=(2, 2), strides=(2, 2),
                         padding="same")

    # Define the model
    image = tf.keras.Input(shape=image_size, name='image')

    # Convolution 1 (510x510)
    x = Conv2D(filters=filters, name='conv_1_1', **conv_params)(image)
    x1 = Conv2D(filters=filters, name='conv_1_2', **conv_params)(x)
    xc1 = MaxPool2D(pool_size=2)(x1)
    x = BatchNormalization()(xc1)
    x = Dropout(0.1)(x)

    # Convolution 2 (253x253)
    x = Conv2D(filters=filters * 2, name='conv_2_1', **conv_params)(x)
    x2 = Conv2D(filters=filters * 2, name='conv_2_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x2)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Convolution 3 (124x124)
    x = Conv2D(filters=filters * 4, name='conv_3_1', **conv_params)(x)
    x3 = Conv2D(filters=filters * 4, name='conv_3_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x3)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Convolution 4 (60x60)
    x = Conv2D(filters=filters * 8, name='conv_4_1', **conv_params)(x)
    x4 = Conv2D(filters=filters * 8, name='conv_4_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x4)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Convolution 5
    x = Conv2D(filters=filters * 16, name='conv_5_1', **conv_params)(x)
    x5 = Conv2D(filters=filters * 16, name='conv_5_2', **conv_params)(x)
    x = MaxPool2D(pool_size=2)(x5)
    x = BatchNormalization()(x)
    x_feats = Dropout(0.1)(x)

    # Attention 1
    x_attn = SelfAttn()(x_feats)

    # Attention 2
    x_attn = Conv2D(filters=filters * 16, name='ConvB', **conv_params)(x_attn)
    x_attn = BatchNormalization()(x_attn)
    x_attn = SelfAttn()(x_attn)

    x_attn = Conv2D(filters=filters * 16, kernel_size=1, padding='same', use_bias=False)(
        tf.concat([x, x_attn], 3))
    x_attn = BatchNormalization()(x_attn)
    x_attn = Dropout(0.1)(x_attn)

    # Deconvolution 1
    x = Conv2DTranspose(filters=filters * 8, name='deconv_1_1', **deconv_params)(x_attn)
    x = Concatenate(axis=-1)([x, x5])

    x = Conv2D(filters=filters * 8, name='deconv_1_2', **conv_params)(x)
    x = Conv2D(filters=filters * 8, name='deconv_1_3', **conv_params)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Deconvolution 2
    x = Conv2DTranspose(filters=filters * 4, name='deconv_2_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x4])

    x = Conv2D(filters=filters * 4, name='deconv_2_2', **conv_params)(x)
    x = Conv2D(filters=filters * 4, name='deconv_2_3', **conv_params)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Deconvolution 3
    x = Conv2DTranspose(filters=filters * 2, name='deconv_3', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x3])

    x = Conv2D(filters=filters * 2, name='deconv_3_2', **conv_params)(x)
    x = Conv2D(filters=filters * 2, name='deconv_3_3', **conv_params)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Deconvolution 4
    x = Conv2DTranspose(filters=filters, name='deconv_4_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x2])

    x = Conv2D(filters=filters, name='deconv_4_2', **conv_params)(x)
    x = Conv2D(filters=filters, name='deconv_4_3', **conv_params)(x)
    x = BatchNormalization()(x)

    # Deconvolution 5
    x = Conv2DTranspose(filters=filters, name='deconv_5_1', **deconv_params)(x)
    x = Concatenate(axis=-1)([x, x1])

    x = Conv2D(filters=filters, name='deconv_5_2', **conv_params)(x)
    x = Conv2D(filters=num_classes, name='deconv_5_3', **conv_params)(x)
    x = BatchNormalization()(x)

    mask = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='mask')(x)

    # Compile the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask], name=name)

    # Compile model
    model.compile(
        loss={'mask': binary_focal_crossentropy},
        optimizer='Adam',
        metrics={'mask': [dice_coef, jaccard_distance, myIOU(), Precision(), Recall()]}
    )

    return model


def callbacks(ts: pd.Timestamp, filepath: Path) -> tuple:
    """
    Define callbacks and store them in a timestamped folder.

    :return:
    """

    # Create folders.
    tb_path = filepath / f'tensorboard/{ts}'
    ckpt_path = filepath / f'checkpoints/{ts}'
    early_stop_path = filepath / f'earlystopping/{ts}'

    ckpt_path.mkdir(parents=True)
    tb_path.mkdir(parents=True)
    early_stop_path.mkdir(parents=True)

    # Define callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path))
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    call_backs = [tensorboard_cb, checkpoint_cb, early_stopping_cb]

    return call_backs, tb_path


def train_model(model, data_pipe, epochs: int, ts, callback_path):
    logger = logging.getLogger(__name__)
    calls, tb_path = callbacks(ts, callback_path)
    logger.log(logging.INFO, 'Callbacks Created')

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(tb_path)])
    url = tb.launch()

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
