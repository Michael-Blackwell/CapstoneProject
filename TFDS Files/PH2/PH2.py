"""PH2 dataset.

Using the TFDS CLI to build this dataset, the data location and output must be specified:
tfds build --data_dir=<Out_Path> --manual_dir=<Data_Parent_Folder>

For more information on this process refer to the documentation:
https://www.tensorflow.org/datasets/add_dataset#write_your_dataset
"""

import tensorflow_datasets as tfds
import pandas as pd
from sklearn.model_selection import train_test_split

# (ISIC2017): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
#PH2 Dataset
"""

# (ISIC2017): BibTeX citation
_CITATION = """
@INPROCEEDINGS{6610779,  
author={Mendonça, Teresa and Ferreira, Pedro M. and Marques, Jorge S. and Marcal, André R. S. and Rozeira, Jorge},  
booktitle={2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},   
title={PH2 - A dermoscopic image database for research and benchmarking},   
year={2013},  
volume={},  
number={},  
pages={5437-5440},  
doi={10.1109/EMBC.2013.6610779}}
"""

# Path to data on personal machine
data_path = "/media/storage/Capstone1/Data/PH2Dataset"


class Ph2(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for PH2 dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download image, mask, and label files from https://www.fc.up.pt/addi/ph2%20database.html and 
    save them in the `manual_dir/`.
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # (Ph2): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'mask': tfds.features.Image(shape=(None, None, 1))
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'mask'),  # Set to `None` to disable
            homepage='https://www.fc.up.pt/addi/ph2%20database.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # (ISIC2017): Downloads the data and defines the splits
        base = dl_manager.manual_dir / 'PH2Dataset'

        labels = pd.read_excel(f'{data_path}/PH2_dataset.xlsx', header=12, usecols=['Image Name', 'Atypical Nevus', 'Melanoma'])
        labels = labels.fillna(0)
        labels = labels.replace(to_replace='X', value=1)
        x_train, x_val, y_train, y_val = train_test_split(labels['Image Name'],
                                                          labels[['Atypical Nevus', 'Melanoma']],
                                                          test_size=0.3,
                                                          random_state=42,
                                                          stratify=labels[['Atypical Nevus', 'Melanoma']])

        x_val1, x_test, y_val1, y_test = train_test_split(labels['Image Name'],
                                                          labels[['Atypical Nevus', 'Melanoma']],
                                                          test_size=0.66,
                                                          random_state=42,
                                                          stratify=labels[['Atypical Nevus', 'Melanoma']])

        train = pd.merge(labels, x_train, how='inner', on='Image Name')
        val = pd.merge(labels, x_val1, how='inner', on='Image Name')
        test = pd.merge(labels, x_test, how='inner', on='Image Name')

        return {
            'train': self._generate_examples(train, base, 'Training'),
            'val': self._generate_examples(val, base, 'Validation'),
            'test': self._generate_examples(test, base, 'Test'),
        }

    def _generate_examples(self, labels, path, split_type):
        """Yields examples."""

        for idx, labs in labels.iterrows():
            yield labs['Image Name'], {
                'image': path / f'PH2 Dataset images/{labs["Image Name"]}/{labs["Image Name"]}_Dermoscopic_Image/{labs["Image Name"]}.bmp',
                'mask': path / f'PH2 Dataset images/{labs["Image Name"]}/{labs["Image Name"]}_lesion/{labs["Image Name"]}_lesion.bmp'
            }
