# -*- coding: utf-8 -*-

from src.utils.paths import PATH_DATA_RAW, PATH_DATA_PROCESSED
import src.utils.input_output as io


class DataLoader:
    # Common data frames
    # datasets
    df_raw = None

    # downloaded raw filenames
    FILE_NAME_RAW = 'creditcard.csv'

    # processed filenames
    FILE_NAME_X_TRAIN = 'X_train.pkl'
    FILE_NAME_X_TEST = 'X_test.pkl'
    FILE_NAME_Y_TRAIN = 'y_train.pkl'
    FILE_NAME_Y_TEST = 'y_test.pkl'
    FILE_NAME_ERROR_DF = 'error_df.pkl'

    # output filenames
    FILE_NAME_MODEL = 'encoder.h5'

    def __init__(self):
        """
        This object deals with all data loading tasks
        """

    def load_raw_data(self):
        return io.read_csv(PATH_DATA_RAW, self.FILE_NAME_RAW)

    def load_autoencoder_model(self):
        return io.read_hdf5(PATH_DATA_PROCESSED, self.FILE_NAME_MODEL)

    def load_X_test(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_X_TEST)

    def load_y_test(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_Y_TEST)

    def load_error_df(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_ERROR_DF)