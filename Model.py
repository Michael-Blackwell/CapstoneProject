from ModelFunctions import *
import tensorflow_addons as tfa
import tensorboard

timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
# width, height, depth
image_shape = (1024, 1024, 3)
train_path = Path('/media/storage/RSNA Brain Tumor Project/train_tr')
val_path = Path('/media/storage/RSNA Brain Tumor Project/val_tr')


def build_model(width, height, depth, name='Model'):
    """Build a 3D convolutional neural network model
    There are many ways to do this."""

    image = tf.keras.Input(shape=(width, height, depth), name='Input')

    kernelsize = (11, 11, 3)
    filters = 32

    # Convolution
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernelsize)(image)
    # f = tf.keras.layers.MaxPool3D(pool_size=3)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)


    # Convolve again
    x = tf.keras.layers.Conv3D(filters=2, kernel_size=1, activation='relu')(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    mask = tf.keras.layers.GlobalMaxPooling2D()(x)
    mask = tf.keras.layers.Activation('softmax')(mask)
    # TODO implement mask output
    # Flatten, Dense, Output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=50, activation="relu")(x)
    melanoma = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    keratosis = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # Define the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask, melanoma, keratosis], name=name)

    # Compile model
    model.compile(
        loss=tfa.losses.sigmoid_focal_crossentropy,
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["binary_accuracy"],
    )

    return model


def train_model(train, batch: int, epochs: int):
    # Build the model
    model = build_model(width=image_shape[0],
                        height=image_shape[1],
                        depth=image_shape[2])

    # Define callbacks and create folders to save them in
    tb_path = Path.cwd() / f'Callbacks/tensorboard/{timestamp}'
    ckpt_path = Path.cwd() / f'Callbacks/checkpoints/{timestamp}'
    early_stop_path = Path.cwd() / f'Callbacks/earlystopping/{timestamp}'

    ckpt_path.mkdir()
    tb_path.mkdir()
    early_stop_path.mkdir()

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_path),
                                                    # write_images=True,
                                                    histogram_freq=1,
                                                    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="binary_accuracy", patience=15)

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(tb_path)])
    url = tb.launch()

    # Training data input generator
    train_gen = DataPipe(filepath=train_path)
    training_data = tf.data.Dataset.from_generator(train_gen.input_generator,
                                                   output_types=((tf.float32, tf.float32, tf.float32, tf.float32),
                                                                 tf.int64),
                                                   output_shapes=((image_shape, image_shape, image_shape, image_shape),
                                                                  (1, 1)))
    # validation data input generator
    val_gen = DataPipe(filepath=val_path)
    validation_data = tf.data.Dataset.from_generator(val_gen.input_generator,
                                                     output_types=((tf.float32, tf.float32, tf.float32, tf.float32),
                                                                   tf.int64),
                                                     output_shapes=(
                                                         (image_shape, image_shape, image_shape, image_shape),
                                                         (1, 1)))

    # Change to 'CPU:0' to use CPU instead of GPU
    with tf.device('GPU:0'):
        model.fit(
            training_data.batch(batch),  # input_generator(train_path, scan_type, 1),#
            validation_data=validation_data.batch(batch),  # input_generator(val_path, scan_type, 1),   #
            epochs=epochs,
            batch_size=batch,
            # shuffle=True,
            # verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
        )

    # save model
    model.save(f'./models/All_Scans{timestamp}')

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
    train_model(train_path, val_path, batch=2, epochs=10)


