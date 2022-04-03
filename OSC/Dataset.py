"""
Author: mike
Created: 3/31/22

"""
import tensorflow as tf
import tensorflow_datasets as tfds


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

        ds = ds.shuffle(buffer_size=100)
        ds = ds.batch(self.batch, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _transform(self, ds):
        """Normalizes images: `uint8` -> `float32`."""
        ds['image'] = tf.image.resize(ds['image'], self.image_size[0:2])
        ds['image'] = tf.image.per_image_standardization(ds['image'])

        if 'mask' in self.output:
            ds['mask'] = tf.image.resize(ds['mask'], self.image_size[0:2])
            ds['mask'] = ds['mask'] / 255

        return ds
