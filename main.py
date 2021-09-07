# This is a sample Python script.

import sys
import time
from src.utils.logger import Logger
from src.data.data_loader import DataLoader
from src.configs.config import Configs
from src.models.train_model import train_model
from src.models.predict_model import model_predict, predict_confusion_matrix



def main(configs: Configs = None, data_loader: DataLoader = None):

    # Load data
    if configs is None:
        configs = Configs('config_function_selection.yaml')

    if data_loader is None:
        data_loader = DataLoader()

    if configs.train_autoencoder_model:
        raw_data = data_loader.load_raw_data()
        train_model(raw_data, configs.test_set_percent, configs.epoch, configs.batch_size)

    if configs.predict_autoencoder_model:
        autoencoder = data_loader.load_autoencoder_model()
        X_test = data_loader.load_X_test()
        y_test = data_loader.load_y_test()
        model_predict(autoencoder, X_test, y_test)

    if configs.generate_predict_confusion_matrix:
        error_df = data_loader.load_error_df()
        predict_confusion_matrix(error_df, configs.reconstruction_error_threshold)



if __name__ == '__main__':
    START = time.time()

    configs = None
    if len(sys.argv) > 1:
        configs = Configs(sys.argv[1])

    main(configs)
    END = time.time()
    Logger.info('Script completed in', '%i seconds' % (END - START), __file__)

