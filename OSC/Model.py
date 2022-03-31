"""
Author: mike
Created: 3/31/22

"""
from ModelFunctions import *
from Functions import *
import tensorboard
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Permute, Softmax, Conv2DTranspose, MaxPool2D, \
    Activation


def INF(B, H, W):
    vec_1d = tf.repeat(tf.constant(float('inf')), repeats=H)
    diag = -tf.linalg.diag(vec_1d)
    diag = tf.reshape(diag, (1, H, H))
    ddiag = tf.tile(diag, (B * W, 1, 1))
    return ddiag


def build_model(image_size, filters=32, kernelsize=3, batch_size=2, recurrence=2, name='CCModel'):

    inter_channels = (filters * 8) // 4
    out_channels = 8
    num_classes = 2
    gamma = 0.05
    cc_imagesize = image_size[0]  # inter_channels  # TODO should be h/w?

    # Input
    image = tf.keras.Input(shape=image_size, name='image', batch_size=batch_size)

    # Convolution
    x = Conv2D(filters=filters, kernel_size=kernelsize, name='conv_1')(image)
    x = MaxPool2D(pool_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # Convolution
    x = Conv2D(filters=filters * 2, kernel_size=kernelsize, name='conv_2')(x)
    x = MaxPool2D(pool_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)


    # Convolution
    x = Conv2D(filters=filters * 4, kernel_size=kernelsize, name='conv_3')(x)
    x = MaxPool2D(pool_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)


    # Convolution
    x = Conv2D(filters=filters * 8, kernel_size=kernelsize, name='conv_4')(x)
    x = MaxPool2D(pool_size=3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Upsample to 1/8 of input image size
    kernels = image_size[0] - 1
    x = Conv2DTranspose(filters=inter_channels, kernel_size=kernels)(x)

    # RCCA
    x_attn = Conv2D(filters=inter_channels, kernel_size=3, padding='same', use_bias=False, name='ConvA')(x)  # TODO padding
    x_attn = BatchNormalization()(x_attn)

    # Criss Cross Attention
    for i in range(recurrence):
        m_batchsize, height, width, channels = x_attn.shape

        # Obtain Query, Key, and Value feature maps
        Query = Conv2D(filters=cc_imagesize // 8, kernel_size=1, name='QueryConv')(x_attn)
        Key = Conv2D(filters=cc_imagesize // 8, kernel_size=1, name='KeyConv')(x_attn)
        Value = Conv2D(filters=cc_imagesize, kernel_size=1, name='ValueConv')(x_attn)

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
        energy_H = tf.keras.layers.Dot(axes=(2, 1))([proj_query_H, proj_key_H]) + temp
        energy_H = tf.reshape(energy_H, (m_batchsize, width, height, height))
        energy_H = Permute((2, 1, 3))(energy_H)
        # W
        energy_W = tf.keras.layers.Dot(axes=(2, 1))([proj_query_W, proj_key_W])
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
        out_H = tf.keras.layers.Dot(axes=(2, 1))([proj_value_H, att_H])
        out_H = tf.reshape(out_H, (m_batchsize, width, -1, height))
        out_H = Permute((3, 1, 2))(out_H)
        # W
        out_W = tf.keras.layers.Dot(axes=(2, 1))([proj_value_W, att_W])
        out_W = tf.reshape(out_W, (m_batchsize, height, -1, width))
        out_W = Permute((1, 3, 2))(out_W)

        x_attn = gamma * (out_H + out_W) + x_attn


    #Output
    x_attn = Conv2D(filters=inter_channels, kernel_size=3, padding='same', use_bias=False, name='ConvB')(x_attn)  # TODO padding
    x_attn = BatchNormalization()(x_attn)

    x_attn = Conv2D(filters=out_channels, kernel_size=3, padding='same', use_bias=False)(tf.concat([x, x_attn], 3))  # TODO padding
    x_attn = BatchNormalization()(x_attn)  # TODO normalize by features axis
    x_attn = Dropout(0.1)(x_attn)
    x_attn = Conv2D(filters=num_classes, kernel_size=1, use_bias=True)(x_attn)

    # x = Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(x)
    mask = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='mask')(x_attn)
    # mask = mask_out(x)

    # TODO Define hybrid (weighted) Loss Function
    # def hybrid_loss(y_true, y_pred): todo focal & binary cross entropy loss 50/50

    # todo metrics -> jaccard/dice & binary accuracy

    # Define the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask], #melanoma, keratosis],
                           name=name)

    # Compile model
    model.compile(
        loss=dice_coef_loss, # tfa.losses.sigmoid_focal_crossentropy,  # 'categorical_crossentropy',  # tfa.losses.sigmoid_focal_crossentropy,
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=[dice_coef, "accuracy"]  # , tfa.metrics.MultiLabelConfusionMatrix],
    )

    return model


def callbacks(ts: pd.Timestamp) -> tuple:
    """
    Define callbacks and store them in a timestamped folder.

    :return:
    """

    # Create folders.
    tb_path = Path.cwd() / f'Callbacks/tensorboard/{ts}'
    ckpt_path = Path.cwd() / f'Callbacks/checkpoints/{ts}'
    early_stop_path = Path.cwd() / f'Callbacks/earlystopping/{ts}'

    ckpt_path.mkdir()
    tb_path.mkdir()
    early_stop_path.mkdir()

    # Define callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path),
                                                    write_images=True,
                                                    histogram_freq=1,
                                                    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # TODO add confusion matrix & jaccard/dice callbacks

    call_backs = [tensorboard_cb, checkpoint_cb, early_stopping_cb]

    return call_backs, tb_path


def train_model(model, data_pipe, epochs: int, ts):

    calls, tb_path = callbacks(ts)

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(tb_path)])
    # url = tb.launch()

    # Change to 'CPU:0' to use CPU instead of GPU
    with tf.device('GPU:0'):
        model.fit(
            data_pipe.train,
            validation_data=data_pipe.val,
            epochs=epochs,
            callbacks=calls
        )

    return model
