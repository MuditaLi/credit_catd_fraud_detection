
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt
from tensorflow import keras


df = pd.read_csv("data/raw/creditcard.csv")
df['Class'].value_counts()
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']


# standardize 'Amount' column
min_amount = min(X['Amount'])
max_amount = max(X['Amount'])
X['Amount_norm'] = (X['Amount'] - min_amount) / (max_amount - min_amount)
X = X.drop(['Amount'], axis=1)


# split train & test
split = StratifiedShuffleSplit(test_size=0.2, random_state=9)
for train_idx, test_idx in split.split(X, y):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# Oversample with SMOTE and random undersample for imbalanced dataset
over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_smote, y_smote = pipeline.fit_resample(X_train, y_train)


# ============================ Build model ================================
input_dim = X_train.shape[1]
# input layer
input_layer = layers.Input(shape=(input_dim, ))
# hidden layers
encoder = layers.Dropout(0.3)(input_layer)
encoder = layers.Dense(units=30, activation="relu")(encoder)
encoder = layers.Dense(units=15, activation="relu")(encoder)
encoder = layers.Dense(units=5, activation="relu")(encoder)
decoder = layers.Dense(units=15, activation="relu")(encoder)
decoder = layers.Dense(units=30, activation="relu")(decoder)
decoder = layers.Dropout(0.3)(decoder)
output = layers.Dense(1, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=output)
autoencoder.summary()
# ============================ Build model ================================
autoencoder = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(X_train.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
autoencoder.summary()



metrics = [ keras.metrics.FalseNegatives(name="fn"),
           keras.metrics.FalsePositives(name="fp"),
           keras.metrics.TrueNegatives(name="tn"),
           keras.metrics.TruePositives(name="tp"),
           keras.metrics.Precision(name="precision"),
           keras.metrics.Recall(name="recall")]

autoencoder.compile(optimizer=keras.optimizers.Adam(1e-2),
                    loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  #  metrics=metrics
                   )

# callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, restore_best_weights=True)
counts = np.bincount(y_train)
weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
class_weight = {0: weight_for_0, 1: weight_for_1}
history = autoencoder.fit(X_train, y_train,
                    batch_size=1024,
                    epochs=100,
                    verbose=2,
                  #  callbacks=callbacks,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight,  # pay more attention to minority class
                    )


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.legend()
plt.show()


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
predictions = autoencoder.predict(X_test)
y_pred = [1 if pred > 0.5 else 0 for pred in predictions]
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# =========================== PLOTS ===============================

# save the encoder to file
# autoencoder.save('encoder.h5')

# error distribution:
predictions = autoencoder.predict(X_test)
mae = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mae,
                        'true_class': y_test})


# ---------------------- ROC doesnt work for unbalanced data ------------------------
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# ------------------------------------------------------------------------------------


# recall & precision
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()

plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.show()
# ------------------------------------------------------------------------------------------------

# reconstruction error: if the error > threshold, then mark as fraud:
LABELS = ["Normal", "Fraud"]
threshold = 5.5
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


def precision_by_threshold(threshold):
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
    return precision

def recall_by_threshold(threshold):
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    return recall

precisions = []
recalls = []
for th in range(0, 20):
    precisions.append(precision_by_threshold(th))
    recalls.append(recall_by_threshold(th))

th = range(0, 20)
plt.plot(th, precisions)
plt.plot(th, recalls)
plt.show()