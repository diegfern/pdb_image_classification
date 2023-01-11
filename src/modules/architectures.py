"""All CNN arquitectures"""
from math import ceil, sqrt
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score,
    f1_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
)

from scipy.stats import (kendalltau, pearsonr, spearmanr)
from keras.utils.layer_utils import count_params

class CnnA(tf.keras.models.Sequential):

    def __init__(self, train_dataset, labels, mode="binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=2))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=4))

        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2,activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=4))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnB(tf.keras.models.Sequential):

    def __init__(self, train_dataset, labels, mode="binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnC(tf.keras.models.Sequential):

    def __init__(self, train_dataset, labels, mode="binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size= 2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
    
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnD(tf.keras.models.Sequential):
    
    def __init__(self, train_dataset, labels, mode="binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size= 2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())

        unit = 64
        while unit > len(labels):
            self.add(tf.keras.layers.Dense(units=unit, activation="tanh"))
            unit = int(unit / 2)

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnE(tf.keras.models.Sequential):
    def __init__(self, train_dataset, labels, dim, mode="binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Resizing( int(dim/2), int(dim/2) ))

        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnF(tf.keras.models.Sequential):
    
    def __init__(self, train_dataset, labels, dim, mode="binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Resizing( int(dim/2), int(dim/2) ))
        
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.2))

        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, ctivation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnG(tf.keras.models.Sequential):
    
    def __init__(self, train_dataset, labels, dim, mode="binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Resizing( int(dim/2), int(dim/2) ))
        
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.2))

        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class CnnH(tf.keras.models.Sequential):
    
    def __init__(self, train_dataset, labels, dim, mode="binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Resizing( int(dim/2), int(dim/2) ))
        
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.2))

        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Flatten())

        unit = 64
        while unit > len(labels):
            self.add(tf.keras.layers.Dense(units=unit, activation="tanh"))
            unit = int(unit / 2)

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1, activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

class Models:
    """Organize CNN objects, train and validation process"""
    def __init__(self, train_dataset, test_dataset, dim, labels, mode, arquitecture):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dim = dim
        self.labels = labels
        self.mode = mode
        self.arquitecture = arquitecture

        if self.arquitecture == "A":
            self.cnn = CnnA(train_dataset=self.train_dataset, labels=self.labels, mode=self.mode)
        elif self.arquitecture == "B":
            self.cnn = CnnB(train_dataset=self.train_dataset, labels=self.labels, mode=self.mode)
        elif self.arquitecture == "C":
            self.cnn = CnnC(train_dataset=self.train_dataset, labels=self.labels, mode=self.mode)
        elif self.arquitecture == "D":
            self.cnn = CnnD(train_dataset=self.train_dataset, labels=self.labels, mode=self.mode)
        elif self.arquitecture == "E":
            self.cnn = CnnE(train_dataset=self.train_dataset, labels=self.labels, dim=self.dim, mode=self.mode)
        elif self.arquitecture == "F":
            self.cnn = CnnF(train_dataset=self.train_dataset, labels=self.labels, dim=self.dim, mode=self.mode)
        elif self.arquitecture == "G":
            self.cnn = CnnG(train_dataset=self.train_dataset, labels=self.labels, dim=self.dim, mode=self.mode)
        elif self.arquitecture == "H":
            self.cnn = CnnH(train_dataset=self.train_dataset, labels=self.labels, dim=self.dim, mode=self.mode)
        else:
            print("Wrong arquitecture for this dataset")
            exit()

    def fit_models(self, epochs, verbose):
        """Fit model"""
        self.cnn.fit(self.train_dataset, epochs=epochs, verbose=verbose, validation_data=self.test_dataset)

    def save_model(self, folder, prefix=""):
        """
        Save model in .h5 format, in 'folder' location
        """
        self.cnn.save(f"{folder}/{prefix}-{self.arquitecture}-{self.mode}.h5")

    def get_metrics(self):
        """
        Returns classification performance metrics.
        Accuracy, recall, precision, f1_score, mcc.
        """
        y_train =  np.array([])
        for _, y in self.train_dataset:
            y_train = np.concatenate([y_train, y.numpy()])

        y_test =  np.array([])
        for _, y in self.test_dataset:
            y_test = np.concatenate([y_test, y.numpy()])
        
        trainable_count = count_params(self.cnn.trainable_weights)
        non_trainable_count = count_params(self.cnn.non_trainable_weights)
        result = {}
        result["arquitecture"] = self.arquitecture
        result["trainable_params"] = trainable_count
        result["non_trainable_params"] = non_trainable_count
        if self.mode == "binary":
            y_train_predicted = np.round_(self.cnn.predict(self.train_dataset))
            y_test_score = self.cnn.predict(self.test_dataset)
            y_test_predicted = np.round_(y_test_score)
        if self.mode == "classification":
            y_train_predicted = np.argmax(self.cnn.predict(self.train_dataset), axis=1)
            y_test_score = self.cnn.predict(self.test_dataset)
            y_test_predicted = np.argmax(y_test_score, axis=1)

        result["labels"] = self.labels.tolist()
        train_metrics = {
            "accuracy": accuracy_score(y_true=y_train, y_pred=y_train_predicted),
            "recall": recall_score(
                y_true=y_train, y_pred=y_train_predicted, average="micro"),
            "precision": precision_score(
                y_true=y_train, y_pred=y_train_predicted, average="micro"),
            "f1_score": f1_score(
                y_true=y_train, y_pred=y_train_predicted, average="micro"),
            "mcc": matthews_corrcoef(y_true=y_train, y_pred=y_train_predicted),
            "confusion_matrix": confusion_matrix(
                y_true=y_train, y_pred=y_train_predicted).tolist()
        }
        test_metrics={
            "accuracy": accuracy_score(y_true=y_test, y_pred=y_test_predicted),
            "recall": recall_score(
                y_true=y_test, y_pred=y_test_predicted, average="micro"),
            "precision": precision_score(
                y_true=y_test, y_pred=y_test_predicted, average="micro"),
            "f1_score": f1_score(
                y_true=y_test, y_pred=y_test_predicted, average="micro"),
            "mcc": matthews_corrcoef(y_true=y_test, y_pred=y_test_predicted),
            "confusion_matrix": confusion_matrix(
                y_true=y_test, y_pred=y_test_predicted).tolist()
        }

        if self.mode == "binary":
            test_metrics["roc_auc_score"] = roc_auc_score(
                y_true=y_test, y_score=y_test_score, average="micro")
        else:
            test_metrics["roc_auc_score"] = roc_auc_score(
                y_true=y_test, y_score=y_test_score, multi_class='ovr')

        result["train_metrics"] = train_metrics
        result["test_metrics"] = test_metrics
        return result
