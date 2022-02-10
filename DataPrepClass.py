from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from pathlib import Path
from multiprocessing.pool import ThreadPool


class DataPrep:
    """Class for preprocessing .dcm / dicom images and saving them in TFR binary files
    with jobs=3 and tfr_file_size=100 runtime is approx. 13 min using Ryzen 7 5800x w/ Nvidia GTX 970

    :param pathlib.Path labels_path:
        filepath to the csv file containing the labels

    :param pathlib.Path data_path:
        filepath to the folder containing the images

    :param pathlib.Path mask_path:
        filepath to the folder containing the image masks

    :param str data_type:
        indicator of which dataset is being prepared, one of ['train', 'val', 'test']

    :param int jobs:
        the number of concurrent jobs that will be run with

    :param tuple image_size:
        (W, H, D) dimensions of the output, input images will be scaled to this size

    :param int tfr_file_size:
        number of images to be saved in a single tfr file

    :return

    """

    def __init__(self,
                 labels_path,
                 data_path,
                 mask_path,
                 data_type,
                 jobs,
                 image_size=(1024, 1024, 3),
                 tfr_file_size=100):

        # Verify dataset type is valid
        datatypes = ['train', 'val', 'test']
        if data_type not in datatypes:
            raise ValueError(f"datatype not recognized. Expected one of {datatypes}")

        self.data_type = data_type
        self.mask_path = mask_path
        self.image_size = image_size
        self.label_path = labels_path
        self.train_data_path = data_path
        self.labels = None
        self.tfr_size = tfr_file_size
        self.output_path = Path.cwd() / f'{data_type}_tr'
        self.output_path.mkdir(exist_ok=True)
        self.jobs = jobs

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte | Taken from TensorFlow documentation."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double. | Taken from TensorFlow documentation."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint | Taken from TensorFlow documentation."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def load_image(self, filepath: str) -> np.ndarray:
        """Load an image, resize it, and normalize the pixel values."""
        image = cv2.imread(filepath)
        image = cv2.resize(image, self.image_size[0:2], interpolation=cv2.INTER_LANCZOS4)  # TODO why interpolate?
        image = image / np.max(image)

        return image

    def get_labels(self) -> pd.DataFrame:
        """Read the labels csv and set the index & dtypes of the fields
        :return
            returns a dataframe of binary labels indexed by image ID."""

        # Create folders for pre-processed training and validation images, replace if existing.
        out = Path.cwd() / self.data_type
        out.mkdir(parents=True, exist_ok=True)

        # Read labels file
        self.labels = pd.read_csv(self.label_path, index_col=['image_id'], dtype={'melanoma': int,
                                                                                  'image_id': str,
                                                                                  'seborrheic_keratosis': int})

        return self.labels

    def write_to_tfr(self, data: list, filename: str) -> None:
        """Loop over list elements (which are dicts) and write each of them to the tfr file."""
        # Compress using Gzip format.
        option = tf.io.TFRecordOptions(compression_type="GZIP")

        # Write the files to the output folder.
        with tf.io.TFRecordWriter(str(self.output_path / f"{filename}.tfrec"), options=option) as writer:
            for element in data:
                # Create the binary TFRecord object and serialize the data into a byte string
                bin_data = tf.train.Example(features=tf.train.Features(feature=element))
                bin_data = bin_data.SerializeToString()
                writer.write(bin_data)

    def preprocess_images(self, divided_labels):
        """Essentially a function call: load, preprocess, & convert images to tensors:
        Output as a serialized TFRecord object (based on protobuf protocol)."""

        tfr_data = []
        label_df, count = divided_labels[0], divided_labels[1]

        for image_id, labels in label_df.iterrows():
            img_data = {}

            # read image and mask files, resizing and pixel normalization takes place upon load.
            img_path = str(self.train_data_path / (image_id + '.jpg'))
            mask_path = str(self.mask_path / (image_id + '_segmentation.png'))
            image = self.load_image(img_path)
            mask = self.load_image(mask_path)

            # Convert mask & image to tensor & serialize
            # inverse operation is tf.io.parse_tensor(image_tensor_ser, out_type=tf.float32)
            image_tensor = tf.convert_to_tensor(image)
            image_tensor_ser = tf.io.serialize_tensor(image_tensor)

            mask_tensor = tf.convert_to_tensor(mask)
            mask_tensor_ser = tf.io.serialize_tensor(mask_tensor)

            # Add features to dict for TFR file
            img_data['image'] = self._bytes_feature(image_tensor_ser)
            img_data['mask'] = self._bytes_feature(mask_tensor_ser)

            bin_id = bytes(image_id, encoding='utf-8')

            img_data['image_width'] = self._int64_feature(self.image_size[0])
            img_data['image_height'] = self._int64_feature(self.image_size[1])
            img_data['image_channels'] = self._int64_feature(self.image_size[2])
            img_data['melanoma_label'] = self._int64_feature(self.labels.loc[image_id, 'melanoma'])
            img_data['keratosis_label'] = self._int64_feature(self.labels.loc[image_id, 'seborrheic_keratosis'])
            img_data['ID'] = self._bytes_feature(bin_id)

            # Add to list of files to write to tfr file
            tfr_data.append(img_data)

        # After all images are loaded, write to tfr file.
        if tfr_data:
            self.write_to_tfr(tfr_data, str(count))

    def data_prep_generator(self):
        """Generator used for multi-threading."""

        shuffled_labels = self.labels.sample(frac=1)
        df = pd.DataFrame(columns=shuffled_labels.columns)
        count = 0

        for idx, data in shuffled_labels.iterrows():
            df.loc[idx] = data
            count += 1

            if len(df) == self.tfr_size:
                yield df, count
                df = pd.DataFrame(columns=shuffled_labels.columns)

        if not df.empty:
            yield df, count

    def compile_tfrecord_files(self) -> None:
        """Concurrently preprocess samples from the dataset into a TFRecord file."""

        self.get_labels()

        # Concurrently process files
        file_total = len(self.labels)/self.tfr_size
        preprocessed_imgs = ThreadPool(self.jobs).imap_unordered(self.preprocess_images, self.data_prep_generator())
        for img_cnt in tqdm(preprocessed_imgs, total=file_total, desc=f'Preprocessing Training Data'):
            pass
