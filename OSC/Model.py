"""
Author: mike
Created: 3/31/22

"""
from Functions import *
import tensorboard
import pandas as pd
from pathlib import Path
from keras.metrics import Precision, Recall, AUC, BinaryIoU
from ModelClass import AttentionUNet
import logging


def compile_model(imag_size: tuple, filters=32, model_name='CCModel'):

    model = AttentionUNet(image_size=imag_size, filters=filters, name=model_name).build_model()

    # Compile model
    model.compile(
        loss={'mask': ['binary_crossentropy']},
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        metrics={'mask': [dice_coef, jaccard_distance, Precision(), Recall(), 'binary_accuracy', AUC(thresholds=[0.5]),
                          BinaryIoU(target_class_ids=[1], threshold=0.5)]}
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
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         mode='auto',
                                                         patience=5,
                                                         restore_best_weights=True)

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
