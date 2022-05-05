import tensorflow as tf
import keras as K


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


def soft_dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice  - Don't round the predictions
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    prediction = K.backend.round(prediction)  # Round to 0 or 1

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def combined_dice_ce_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return 0.5 * dice_coef(target, prediction, axis, smooth) + \
           (1 - 0.5) * K.losses.binary_crossentropy(target, prediction)


class myIOU(tf.keras.metrics.MeanIoU):
    def __init__(self, name, dtype, num_classes):
        super().__init__(num_classes=2)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.backend.round(y_pred)
        return super().update_state(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        return base_config
