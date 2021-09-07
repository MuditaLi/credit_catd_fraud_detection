import numpy as np
import matplotlib.pyplot as plt
from os import path
from src.utils.logger import Logger
from src.utils.paths import PATH_DOCS, PATH_DATA_PROCESSED, PATH_DATA_OUTPUT
import src.utils.input_output as io
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


def read_model_info():
    Logger.info('Get columns')
    feature_selection = 'feature_selection.yaml'
    features = io.read_yaml(PATH_DOCS, feature_selection)
    model_variables = features['model_variables']
    return model_variables


def dataset_standardization(data):
    """
    Since the dataset features are PCA values, we do not need to normalize them, just amount value.
    """

    min_amount = min(data['Amount'])
    max_amount = max(data['Amount'])
    data['Amount_norm'] = (data['Amount'] - min_amount) / (max_amount - min_amount)
    return data


def split_train_test(X, y, test_set_percent):
    """
    Stratified randomized folds preserving the % of samples for each class
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_set_percent, random_state=9)
    for train_idx, test_idx in split.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    return X_train, y_train, X_test, y_test


def SMOTE_under_sampling(X, y, minority_percent, majority_percent_more):
    """
    Oversample with SMOTE and random undersampling for imbalanced dataset
    """

    over = SMOTE(sampling_strategy=minority_percent)
    under = RandomUnderSampler(sampling_strategy=majority_percent_more)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_smote, y_smote = pipeline.fit_resample(X, y)
    return X_smote, y_smote


def autoencoder_model(data):
    """
    Build up autoencoder model structure
    """

    input_dim = data.shape[1]
    input_layer = layers.Input(shape=[input_dim])
    # encoding
    encoder = layers.Dropout(0.1)(input_layer)
    encoder = layers.Dense(30, activation="relu")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.Dense(20, activation="relu")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.Dense(5, activation="relu")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.LeakyReLU()(encoder)
    # decoding
    decoder = layers.Dense(20, activation="relu")(encoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.Dense(30, activation="relu")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.Dropout(0.1)(decoder)
    output = layers.Dense(input_dim, activation='sigmoid')(decoder)
    # define autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def loss_plot(fitted_model):
    """
    Plot train & test loss plots.
    """

    plt.plot(fitted_model.history["loss"], label="Training Loss")
    plt.plot(fitted_model.history["val_loss"], label="Test Loss")
    plt.legend()
    plot_path = path.join(PATH_DATA_OUTPUT, 'loss_plot.png')
    plt.savefig(plot_path)


def train_model(data, test_set_percent=0.2, epoch=200, batch_size=32):
    """
    Train autoencoder
    """

    # select variables for training:
    model_variables = read_model_info()
    Logger.info('Start generating training data')

    data = dataset_standardization(data)
    X = data[model_variables].copy()
    y = data['Class']

    # split dataset into test and train
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_set_percent)

    # same split datasets into processed folder
    io.write_pkl(X_train, path.join(PATH_DATA_PROCESSED, 'X_train.pkl'))
    io.write_pkl(y_train, path.join(PATH_DATA_PROCESSED, 'y_train.pkl'))
    io.write_pkl(X_test, path.join(PATH_DATA_PROCESSED, 'X_test.pkl'))
    io.write_pkl(y_test, path.join(PATH_DATA_PROCESSED, 'y_test.pkl'))

    # build model
    autoencoder = autoencoder_model(X_train)

    # fit the autoencoder model to reconstruct input
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, restore_best_weights=True)
    history = autoencoder.fit(X_train, X_train,
                              epochs=epoch,
                              batch_size=batch_size,
                              verbose=2,
                              validation_data=(X_test, X_test),
                              shuffle=True,
                              callbacks=[early_stop]
                              )

    file_path = path.join(PATH_DATA_PROCESSED, 'encoder.h5')
    autoencoder.save(file_path, overwrite=True)

    # save loss plot
    loss_plot(history)

    return autoencoder




