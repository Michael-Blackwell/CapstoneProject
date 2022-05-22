"""

"""
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


class DataPipe:

    def __init__(self,
                 dataset_name='isic2017',
                 dataset_path='/media/storage/Datasets',
                 batch=16,
                 image_size=(512, 512, 3),
                 output=('labels', 'masks')):
        (self.train, self.val, self.test), self._info = tfds.load(name=dataset_name,
                                                                  data_dir=str(dataset_path),
                                                                  as_supervised=False,
                                                                  download=False,
                                                                  split=['train', 'val', 'test'],
                                                                  with_info=True)

        self.image_size = image_size
        self.mask_size = (image_size[0], image_size[1], 1)
        self.batch = batch
        self.output = output

    def visualize(self, ds, num, image_type):
        """Visualize some examples, type can be image or mask."""
        fig = tfds.show_examples(ds.take(num), self._info, image_key=image_type)

    def transform_all(self):
        """Apply transformations to training, test, and validation datasets."""
        self.train = self.apply_transformations(self.train)
        self.test = self.apply_transformations(self.test)
        self.val = self.apply_transformations(self.val)

    def apply_transformations(self, ds):
        """Transform a Dataset: https://www.tensorflow.org/datasets/keras_example#build_a_training_pipeline"""
        ds = ds.map(self._transform,
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()

        if 'mask' in self.output and 'label' in self.output:
            ds = ds.map(lambda ex: ({'image': ex['image']},
                                    {'mask': ex['mask'],
                                     'melanoma_label': ex['melanoma_label'],
                                     'keratosis_label': ex['keratosis_label']}))
        elif 'mask' in self.output:
            ds = ds.map(lambda ex: ({'image': ex['image']},
                                    {'mask': ex['mask']}))
        elif 'label' in self.output:
            ds = ds.map(lambda ex: ({'image': ex['image']},
                                    {'melanoma_label': ex['melanoma_label'],
                                     'keratosis_label': ex['keratosis_label']}))

        # ds = ds.shuffle(buffer_size=100)
        ds = ds.batch(self.batch, drop_remainder=True) # TODO might not need to drop remainder if batch flags fix NONE
        # ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _transform(self, ds):
        """Normalizes images: `uint8` -> `float32`."""
        ds['image'] = self._resize_img(ds['image'])
        ds['image'] = self._normalize_img(ds['image'])

        if 'mask' in self.output:
            ds['mask'] = self._resize_img(ds['mask'])
            ds['mask'] = self._normalize_img(ds['mask'])

        return ds

    def _resize_img(self, image):
        """Resizes images to specified shape."""
        # return tf.image.resize(image,
        #                        size=self.image_size[0:2],
        #                        method='bicubic',
        #                        preserve_aspect_ratio=False,
        #                        antialias=False,
        #                        name='Resize'
        #                        )
        return tf.image.resize(image, self.image_size[0:2], preserve_aspect_ratio=False)

    @staticmethod
    def _normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.


# def display_sample(display_list):
#     """Show side-by-side an input image,
#     the ground truth and the prediction.
#     """
#     plt.figure(figsize=(18, 18))
#
#     title = ['Input Image', 'True Mask', 'Predicted Mask']
#
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         plt.axis('off')
#     plt.show()


def display_sample(display_list, out_path):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """

    image = tf.keras.preprocessing.image.array_to_img(display_list[0]).convert('L')
    mask_true = tf.keras.preprocessing.image.array_to_img(display_list[1])
    mask_pred = tf.keras.preprocessing.image.array_to_img(display_list[2])

    image = np.array(image)
    mask_true = np.array(mask_true, dtype=bool)
    mask_pred = np.array(mask_pred, dtype=bool)

    jac = jaccard_distance_pred(display_list[1], display_list[2]).numpy()
    jac = round(float(jac), 2)

    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.imshow(mask_true, cmap='gray', interpolation='none', alpha=0.2)
    plt.imshow(mask_pred, cmap='magma', interpolation='none', alpha=0.25)
    plt.axis('off')
    plt.title(f'IoU: {jac}')
    plt.show()

    plt.savefig(str(out_path / f'Attention_{jac}.png'))


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show_predictions(model, out_path, dataset=None, num=(1, 2)):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """

    if dataset:
        count = 0
        for image, mask in dataset.take(num[1]):
            count += 1
            if count < num[0]:
                continue
            pred_mask = model.predict(image)[0]
            pred_mask[pred_mask > 0.5] = 1
            pred_mask[pred_mask <= 0.5] = 0

            display_sample([image['image'][0], mask['mask'][0], pred_mask], out_path)
    # else:
    #     # The model is expecting a tensor of the size
    #     # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
    #     # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
    #     # and we want only 1 inference to be faster
    #     # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
    #     one_img_batch = sample_image[0][tf.newaxis, ...]
    #     # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
    #     inference = model.predict(one_img_batch)
    #     # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
    #     pred_mask = create_mask(inference)
    #     # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
    #     display_sample([sample_image[0], sample_mask[0],
    #                     pred_mask[0]])

# if __name__ == '__main__':
#     test = DataPipe()
#     test.transform_all()
#     test.visualize(test.val, 4, 'mask')


@tf.function
def jaccard_loss(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return tf.reduce_mean(jd)


@tf.function
def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance"""
    y_pred = K.backend.round(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = jac * smooth
    return tf.reduce_mean(jd)


def jaccard_distance_pred(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance"""
    y_pred = K.backend.round(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    jac = intersection / (sum_ - intersection)
    jd = jac * smooth
    return tf.reduce_mean(jd)


@tf.function
def dice_coef_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    return dice_loss


@tf.function
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


class myIOU(tf.keras.metrics.MeanIoU):
    def __init__(self, name, dtype, num_classes):
        super().__init__(num_classes=2)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.backend.round(y_pred)
        return super().update_state(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        return base_config
