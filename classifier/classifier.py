#!/usr/bin/env python
"""
A base class for the different classifiers
"""
import time
from abc import ABC, abstractmethod

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
import tensorflow as tf

from classifier.hyperparameters import Hyperparameters
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import save_logs


class Classifier(ABC):
    def __init__(self, output_directory, input_shapes, nb_classes, verbose=True, hyperparameters=None,
                 model_init=None):
        self.output_directory = output_directory
        self.verbose = verbose
        self.callbacks = []
        self.hyperparameters = hyperparameters

        self.model = model_init if model_init else self.build_model(input_shapes, nb_classes, hyperparameters)
        self.create_callbacks()

    @abstractmethod
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        pass

    def create_callbacks(self):
        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.callbacks.append(model_checkpoint)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.hyperparameters.reduce_lr_factor,
                                      patience=self.hyperparameters.reduce_lr_patience)
        self.callbacks.append(reduce_lr)

        early_stopping = EarlyStopping(patience=15)
        self.callbacks.append(early_stopping)

    def get_optimizer(self):
        return Adam(lr=self.hyperparameters.lr, decay=self.hyperparameters.decay)

    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=16, nb_epochs=5000, x_test=None, shuffle=True):
        mini_batch_size = int(min(x_train[0].shape[0] / 10, batch_size))
        class_labels = np.argmax(y_train, axis=1)
        print(class_labels.shape)
        class_weights = compute_class_weight('balanced', np.unique(class_labels), class_labels)

        GLOBAL_LOGGER.info("Batch size: {}".format(batch_size))
        GLOBAL_LOGGER.info("Mini batch size: {}".format(mini_batch_size))
        GLOBAL_LOGGER.info("Number of epochs: {}".format(nb_epochs))
        GLOBAL_LOGGER.info("Fitting model")
        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, verbose=2,
                              validation_data=(x_val, y_val), class_weight=class_weights, callbacks=self.callbacks, shuffle=shuffle)
        duration = time.time() - start_time
        
        GLOBAL_LOGGER.info(f"Loading weights and predicting")
        
        self.model.save(self.output_directory + 'best_model_last_epoch.h5')
        self.model.load_weights(self.output_directory + 'best_model.h5')
        y_pred_probabilities = self.model.predict(x_test if x_test else x_val)

        y_pred = np.argmax(y_pred_probabilities, axis=1)

        return save_logs(self.output_directory, hist, y_pred, y_pred_probabilities, y_true, duration)


def get_multipliers(channels_no, hyperparameters: Hyperparameters):
    filters_multipliers = [1] * channels_no
    kernel_size_multipliers = [1] * channels_no

    if hyperparameters:
        filters_multipliers = hyperparameters.filters_multipliers
        kernel_size_multipliers = hyperparameters.kernel_size_multipliers

    return filters_multipliers, kernel_size_multipliers

def reshape_samples(samples):
    return [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in samples]

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed
