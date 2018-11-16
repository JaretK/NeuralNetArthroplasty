"""
Utility functions for use in neural network construction and processing
"""

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras import initializers
import keras.backend as K
from keras.constraints import maxnorm
from keras.layers import Layer
import tensorflow as tf
import logging
import matplotlib; matplotlib.use('Agg') # allow backend matplotlib without setting environment variables
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

####################
# 
# Global variables for model / backprop
# 

EPOCHS = 300 #300 standard number of epochs BEFORE stopping early
LEARNING_RATE = 0.001 # standard learning rate
BATCH_SIZE = 200 
VALIDATION_SPLIT = 0.1 # for 9:1 stratified K fold
EPSILON = 0.01 
VERBOSE_LEVEL = 0 # not verbose
DROPOUT_RATE = 0.85 # dropout NOT used, here for future studies
MODEL_ARCHITECTURE = [112, 56, 28, 14]

####################
# 
# Final nodes for regression models. All classification models use softmax
# 

def fit_model(model, train_data_x, train_data_y):
    """
    Runs the model using the train x and train y data

    Performs monitoring to stop the learning early if validation loss does not improve
    """
    log = logging.getLogger('model.fit')
    log.info('model.fit')
    early_stop = EarlyStopping(monitor='val_loss', patience=20)
    log.info('model currently fitting. This might take a while...')
    history = model.fit(train_data_x, train_data_y,
                        epochs = EPOCHS,
                        validation_split = VALIDATION_SPLIT,
                        batch_size = BATCH_SIZE,
                        verbose = VERBOSE_LEVEL,
                        callbacks = [early_stop,PrintDot()])
    return history

def get_base_optimizer():
    """
    Base optimizer. Adagrad was chosen for its stochastic properties
    """
    return tf.train.AdagradOptimizer(
                LEARNING_RATE
    )

def get_base_initializer():
    """
    Base initializer 
    """
    return initializers.glorot_normal()

def get_base_regularizer():
    '''
    TODO
    Base regularizer. Not yet implemented
    '''
    return 

def build_base_model(input_length, round=False):
    """
    Builds a base model with nodes derived from a typical DNN for regression
    """
    log = logging.getLogger('build_base_model')
    log.info('build_base_model')
    model = Sequential([
        Dense(input_length+1, kernel_initializer = get_base_initializer(), activation=tf.nn.relu,input_shape=(input_length,)),
        Dense(input_length/2, kernel_initalizer = get_base_initializer(), activation=tf.nn.relu),
        Dense(1, activation='linear')
        ])
    if round:
        model.add(Round())
    optimizer = get_base_optimizer()
    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics=['mae'])
    return model

def build_shallow_model(input_length, round=False):
    """
    Shallow model that approximates a linear regressor
    """
    log = logging.getLogger('build_shallow_model')
    log.info('build_shallow_model')
    model = Sequential([
        Dense(input_length+1, kernel_initializer = get_base_initializer(), activation=tf.nn.relu,input_shape=(input_length,)),
        Dense(1, activation='linear')
        ])
    if round:
        model.add(Round())
    optimizer = get_base_optimizer()
    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics=['mae'])
    return model

def build_abstract_softmax_classification_model(input_length, nodes, num_classes):
    """
    Abstracted model that is user-defined based on the nodes parameter

    For classifying to num_classes of classes
    """
    log = logging.getLogger('build_abstract_softmax_classification_model')
    log.info('build_abstract_softmax_classification_model. Classes = %s' % num_classes)
    node_list = [
        Dense(input_length+1,
              activation=tf.nn.relu,
              input_shape=(input_length,),
              kernel_constraint=maxnorm(3))
        ]
    node_list.extend([Dense(x, kernel_initializer = get_base_initializer(),activation=tf.nn.relu, kernel_constraint=maxnorm(3)) for x in nodes])
    model = Sequential(node_list)
    model.add(Dense(int(num_classes), activation=tf.nn.softmax))
    optimizer = get_base_optimizer()
    model.compile(optimizer = optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_abstract_model(input_length, nodes, round=False):
    """
    Abstracted model that is user-defined based on the nodes parameter
    """
    log = logging.getLogger('build_abstract_model')
    log.info('build_abstract_model')
    node_list = [
        Dense(input_length+1,
              activation=tf.nn.relu,
              input_shape=(input_length,))
        ]
    node_list.extend([Dense(x, kernel_initializer = get_base_initializer(), activation=tf.nn.relu) for x in nodes])
    model = Sequential(node_list)
    if round:
        model.add(Round())
    else:
        model.add(Dense(1, activation='linear'))
    optimizer = get_base_optimizer()
    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics=['mae'])
    return model

def plot_epoch_history(history, y_label = "y_units", title = 'EpochHistory.png'):
    """
    Plots training and validation losses by epoch count
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import chain
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    mae = np.array(history.history['acc'])
    val_mae = np.array(history.history['loss'])
    y_max = np.ceil(max(chain(mae, val_mae)))
    plt.plot(history.epoch, mae,
           label='Train accuracy')
    plt.plot(history.epoch, val_mae,
           label = 'Train loss')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(title)

def save_data(name, dataframe):
    """
    Saves a pandas dataframe
    """
    try:
        dataframe.to_csv(name, index=False)
        return True
    except:
        return False

class PrintDot(Callback):
    """Container class for tensorflow model printing to stdout"""
    def on_epoch_end(self, epoch, logs):
        from sys import stdout
        # write a '.' for every 100 epochs, print a new line
        if epoch % 100 == 0:
            stdout.write('\n')
        stdout.write('.')

class Round(Layer):
    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_micro_auc(y_true, y_probas, classes):
    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true))
    y_probas = np.concatenate(y_probas).ravel().tolist()
    fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), y_probas)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

if __name__ == '__main__':
    pass