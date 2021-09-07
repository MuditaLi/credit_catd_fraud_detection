import numpy as np
import pandas as pd
import src.utils.input_output as io
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from src.utils.paths import PATH_DATA_PROCESSED, PATH_DATA_OUTPUT
from sklearn.metrics import (confusion_matrix, precision_recall_curve)


def model_predict(model, X_test, y_test):
    """
    Model prediction & generate precision recall plot
    """

    predictions = model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                             'true_class': y_test})

    file_path = path.join(PATH_DATA_PROCESSED, 'error_df.pkl')
    io.write_pkl(error_df, file_path)

    # generate precision & recall plots
    precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)

    plt.plot(recall, precision, 'b', label='Precision-Recall curve')
    plt.title('Recall vs Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plot_path = path.join(PATH_DATA_OUTPUT, 'recall_vs_precision.png')
    plt.savefig(plot_path)

 #   plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
 #   plt.title('Precision for different threshold values')
 #   plt.xlabel('Threshold')
 #   plt.ylabel('Precision')
 #   plot_path = path.join(PATH_DATA_OUTPUT, 'precision_by_th.png')
 #   plt.savefig(plot_path)

 #   plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
 #   plt.title('Recall for different threshold values')
 #   plt.xlabel('Threshold')
 #   plt.ylabel('Recall')
 #   plot_path = path.join(PATH_DATA_OUTPUT, 'recall_by_th.png')
 #   plt.savefig(plot_path)


def predict_confusion_matrix(data, threshold):
    """
    Apply threshold and generate confusion matrix.
    """
    y_pred = [1 if e > threshold else 0 for e in data.reconstruction_error.values]
    conf_matrix = confusion_matrix(data.true_class, y_pred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"], annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plot_path = path.join(PATH_DATA_OUTPUT, 'predicted_confusion_matrix.png')
    plt.savefig(plot_path)
