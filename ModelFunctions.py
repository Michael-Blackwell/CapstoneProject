"""

"""
import tensorflow as tf
import tensorflow_datasets as tfds


class ModelDataset:

    def __init__(self,
                 dataset_name='isic2017',
                 dataset_path='/media/storage/Datasets',
                 batch=16,
                 image_size=(1024, 1024, 3)):

        (self.train, self.val, self.test), self._info = tfds.load(name=dataset_name,
                                                                  data_dir=str(dataset_path),
                                                                  as_supervised=False,
                                                                  download=False,
                                                                  split=['train', 'val', 'test'],
                                                                  with_info=True)

        self.image_size = image_size
        self.batch = batch

    def visualize(self, ds, num, image_type):
        """Visualize some examples, type can be image or mask."""
        fig = tfds.show_examples(ds.take(num), self._info, image_key=image_type)

    def transform_all(self):
        """Apply transformations to training, test, and validation datasets."""
        self.train = self.apply_transformations(self.train, 'train')
        self.test = self.apply_transformations(self.test, 'test')
        self.val = self.apply_transformations(self.val, 'val')

    def apply_transformations(self, ds, split):
        """Transform a Dataset: https://www.tensorflow.org/datasets/keras_example#build_a_training_pipeline"""
        ds = ds.map(self._transform,
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(self._info.splits[split].num_examples)
        ds = ds.batch(self.batch)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _transform(self, ds):
        """Normalizes images: `uint8` -> `float32`."""
        ds['image'] = self._resize_img(ds['image'])
        ds['image'] = self._normalize_img(ds['image'])

        ds['mask'] = self._resize_img(ds['mask'])
        ds['mask'] = self._normalize_img(ds['mask'])

        return ds

    def _resize_img(self, image):
        """Resizes images to specified shape."""
        return tf.image.resize(image,
                               size=self.image_size[0:2],
                               method='bicubic',
                               preserve_aspect_ratio=False,
                               antialias=False,
                               name='Resize'
                               )

    @staticmethod
    def _normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.


# if __name__ == '__main__':
#     test = ModelDataset()
#     test.transform_all()
#     test.visualize(test.test, 4, 'image')
