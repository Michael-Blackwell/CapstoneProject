"""ISIC2017 dataset.

Using the TFDS CLI to build this dataset, the data location and output must be specified:
tfds build --data_dir=<Out_Path> --manual_dir=<Data_Parent_Folder>

For more information on this process refer to the documentation:
https://www.tensorflow.org/datasets/add_dataset#write_your_dataset
"""

import tensorflow_datasets as tfds
import pandas as pd

# (ISIC2017): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
# ISIC 2017 Challenge Dataset
"""

# (ISIC2017): BibTeX citation
_CITATION = """
@article{DBLP:journals/corr/abs-1710-05006,
  author    = {Noel C. F. Codella and
               David A. Gutman and
               M. Emre Celebi and
               Brian Helba and
               Michael A. Marchetti and
               Stephen W. Dusza and
               Aadi Kalloo and
               Konstantinos Liopyris and
               Nabin K. Mishra and
               Harald Kittler and
               Allan Halpern},
  title     = {Skin Lesion Analysis Toward Melanoma Detection: {A} Challenge at the
               2017 International Symposium on Biomedical Imaging (ISBI), Hosted
               by the International Skin Imaging Collaboration {(ISIC)}},
  journal   = {CoRR},
  volume    = {abs/1710.05006},
  year      = {2017},
  url       = {http://arxiv.org/abs/1710.05006},
  eprinttype = {arXiv},
  eprint    = {1710.05006},
  timestamp = {Tue, 29 Jun 2021 15:47:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1710-05006.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# Path to data on personal machine
data_path = "/media/storage/Capstone1/Data/ISIC_2017"


class Isic2017(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ISIC2017 dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download from image, mask, and label files from https://challenge.isic-archive.com/data/#2017 and 
    save them in the `manual_dir/`.
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # (ISIC2017): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'mask': tfds.features.Image(shape=(None, None, 3)),
                'melanoma_label': tfds.features.ClassLabel(num_classes=2),
                'keratosis_label': tfds.features.ClassLabel(num_classes=2),
                # 'ID': tfds.features.Text(),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', ('mask', 'melanoma_label', 'keratosis_label')),  # Set to `None` to disable
            homepage='https://challenge.isic-archive.com/landing/2017/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # (ISIC2017): Downloads the data and defines the splits
        base = dl_manager.manual_dir / 'ISIC_2017'

        train_lab_path = f'{data_path}/ISIC-2017_Training_Part3_GroundTruth.csv'
        val_lab_path = f'{data_path}/ISIC-2017_Validation_Part3_GroundTruth.csv'
        test_lab_path = f'{data_path}/ISIC-2017_Test_v2_Part3_GroundTruth.csv'
        # image_path = base / 'ISIC-2017_Training_Data'
        # mask_path = base / 'ISIC-2017_Training_Part1_GroundTruth'

        # (ISIC2017): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(train_lab_path, base, 'Training'),
            'val': self._generate_examples(val_lab_path, base, 'Validation'),
            'test': self._generate_examples(test_lab_path, base, 'Test'),
        }

    def _generate_examples(self, label_path, path, split_type):
        """Yields examples."""
        # (ISIC2017): Yields (key, example) tuples from the dataset
        labels = pd.read_csv(label_path, index_col=['image_id'], dtype={'melanoma': int,
                                                                        'image_id': str,
                                                                        'seborrheic_keratosis': int})

        for pat_id, labs in labels.iterrows():
            yield pat_id, {
                'image': path / f'ISIC-2017_{split_type}_Data/{pat_id}.jpg',
                'mask': path / f'ISIC-2017_{split_type}_Part1_GroundTruth/{pat_id}_segmentation.png',
                'melanoma_label': labs[0],
                'keratosis_label': labs[1]
            }
