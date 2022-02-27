from ModelFunctions import *
import tensorflow_addons as tfa
import tensorboard

timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')


def build_model(image_size, name='Model'):
    """Build a 3D convolutional neural network model
    There are many ways to do this."""

    image = tf.keras.Input(shape=image_size, name='image')

    kernelsize = 1
    filters = 32

    # Convolution
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernelsize)(image)
    # f = tf.keras.layers.MaxPool3D(pool_size=3)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Convolve again
    x = tf.keras.layers.Conv2D(filters=2, kernel_size=1, activation='relu')(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    mask = tf.keras.layers.GlobalMaxPooling2D()(x)
    mask = tf.keras.layers.Activation('softmax', name='mask')(mask)
    # TODO implement mask output
    # Flatten, Dense, Output
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(units=50, activation="relu")(x)
    # melanoma = tf.keras.layers.Dense(units=1, activation='sigmoid', name='melanoma_label')(x)
    # keratosis = tf.keras.layers.Dense(units=1, activation='sigmoid', name='keratosis_label')(x)

    # Define the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask], name=name) #melanoma, keratosis], name=name)

    # Compile model
    model.compile(
        loss='categorical_crossentropy',  # tfa.losses.sigmoid_focal_crossentropy,
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
    dataset = DataPipe(batch=5)
    dataset.transform_all()
    train_model(dataset, epochs=10)


