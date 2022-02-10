from DataPrepClass import *


if __name__ == '__main__':

    labels_filepath = Path.cwd() / 'Data/ISIC_2017/ISIC-2017_Training_Part3_GroundTruth.csv'
    data_filepath = Path.cwd() / 'Data/ISIC_2017/ISIC-2017_Training_Data'
    mask_filepath = Path.cwd() / 'Data/ISIC_2017/ISIC-2017_Training_Part1_GroundTruth'
    datatype = 'train'
    img_size = (1024, 1024, 3)
    file_size = 100
    jobs = 3

    pre = DataPrep(labels_path=labels_filepath,
                   data_path=data_filepath,
                   mask_path=mask_filepath,
                   data_type=datatype,
                   image_size=img_size,
                   tfr_file_size=file_size,
                   jobs=jobs)

    pre.get_labels()
    pre.compile_tfrecord_files()
