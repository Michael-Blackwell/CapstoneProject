from ModelFunctions import *
import tensorflow_addons as tfa
import tensorboard
import tensorflow.keras.backend as K
from typing import Callable
"""
Focal Loss:
https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
"""

timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')


def binary_focal_loss(beta: float, gamma: float = 2.) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Focal loss is derived from balanced cross entropy, where focal loss adds an extra focus on hard examples in the
    dataset:
        FL(p, p̂) = −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]
    When γ = 0, we obtain balanced cross entropy.
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Used as loss function for binary image segmentation with one-hot encoded masks.
    :param beta: Weight coefficient (float)
    :param gamma: Focusing parameter, γ ≥ 0 (float, default=2.)
    :return: Focal loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the focal loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Focal loss (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        f_loss = beta * (1 - y_pred) ** gamma * y_true * K.log(y_pred)  # β*(1-p̂)ᵞ*p*log(p̂)
        f_loss += (1 - beta) * y_pred ** gamma * (1 - y_true) * K.log(1 - y_pred)  # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
        f_loss = -f_loss  # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(f_loss))
        f_loss = K.mean(f_loss, axis=axis_to_reduce)

        return f_loss

    return loss


def build_model(image_size, name='Model'):
    """Build a 3D convolutional neural network model
    There are many ways to do this."""

    """Build a 3D convolutional neural network model
        There are many ways to do this."""

    image = tf.keras.Input(shape=image_size, name='image')

    kernelsize = 5
    filters = 100

    # Convolution
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernelsize, padding='same')(image)
    # f = tf.keras.layers.MaxPool3D(pool_size=3)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Convolve again
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    mask = tf.keras.layers.GlobalMaxPooling2D()(x)
    mask = tf.keras.layers.Activation('softmax', name='mask')(mask)

    # Flatten, Dense, Output
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(units=50, activation="relu")(x)
    # melanoma = tf.keras.layers.Dense(units=1, activation='sigmoid', name='melanoma_label')(x)
    # keratosis = tf.keras.layers.Dense(units=1, activation='sigmoid', name='keratosis_label')(x)

    # Define the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask], name=name)  # melanoma, keratosis], name=name)

    # Compile model
    model.compile(
        loss=tfa.losses.sigmoid_focal_crossentropy,  # 'categorical_crossentropy',  # tfa.losses.sigmoid_focal_crossentropy,
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["accuracy"],
    )

    return model


def train_model(data_pipe, epochs: int):
    # Build the model
    model = build_model(data_pipe.image_size)

    # Define callbacks and create folders to save them in
    tb_path = Path.cwd() / f'Callbacks/tensorboard/{timestamp}'
    ckpt_path = Path.cwd() / f'Callbacks/checkpoints/{timestamp}'
    early_stop_path = Path.cwd() / f'Callbacks/earlystopping/{timestamp}'

    ckpt_path.mkdir()
    tb_path.mkdir()
    early_stop_path.mkdir()

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path),
                                                    write_images=True,
                                                    histogram_freq=1,
                                                    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(tb_path)])
    url = tb.launch()

    # Change to 'CPU:0' to use CPU instead of GPU
    with tf.device('GPU:0'):
        model.fit(
            data_pipe.train,  # input_generator(train_path, scan_type, 1),#
            validation_data=data_pipe.val,  # input_generator(val_path, scan_type, 1),   #
            epochs=epochs,
            # shuffle=True,
            # verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
        )

    # save model
    model.save(f'./models/All_Scans{timestamp}')
    # TODO run and save metrics
    # TODO run test data

    # show metrics
    # fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    # ax = ax.ravel()

    # for i, metric in enumerate(["acc", "loss"]):
    #     ax[i].plot(model.history.history[metric])
    #     ax[i].plot(model.history.history["val_" + metric])
    #     ax[i].set_title("{} Model {}".format(scan_type, metric))
    #     ax[i].set_xlabel("epochs")
    #     ax[i].set_ylabel(metric)
    #     ax[i].legend(["train", "val"])

    return model


if __name__ == '__main__':
    dataset = DataPipe(batch=1)
    dataset.transform_all()
    train_model(dataset, epochs=10)


