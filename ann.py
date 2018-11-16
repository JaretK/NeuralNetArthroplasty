#!/usr/bin/env python
# built for python3

import logging
import sys
from datetime import datetime
from time import time
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import keras.backend as K
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 10, shuffle=True)

# ------------------
# Custom imports
import Ann_utilities
from Attributes import Attributes
from Dataframe_utilities import (load_data,
                                 get_x,
                                 normalize_series,
                                 DataframeProcessor)

def train_model(model, train_data_x, train_data_y):
    """
    Trains the model object using train_data_x and train_data_y
    """
    log = logging.getLogger('model.train')
    start_time = float(time())
    log.info('Start training model')
    print(model.summary())
    history = Ann_utilities.fit_model(model, train_data_x,train_data_y)
    log.info('Model trained in %fs' % (float(time()) - start_time))
    return model, history

def test_model(model, test_data_x, test_data_y):
    log = logging.getLogger('model.test')
    log.info('Start testing model')
    start_time = float(time())
    [loss, metric] = model.evaluate(test_data_x, test_data_y, verbose=1)
    log.info("Loss: %f" % loss)
    log.info("Testing set Mean Abs Error / accuracy metric: %f" % (metric))
    test_prediction_probas = model.predict(test_data_x)
    test_predicted_labels = [np.argmax(x) for x in test_prediction_probas]
    return loss, metric, test_prediction_probas, test_predicted_labels

def evaluate_model(model, train_x, train_y, test_x, test_y):
    log = logging.getLogger('model.evaluate')
    log.info('model.evaluate')
    model, train_history = train_model(model, train_x, train_y)
    Ann_utilities.plot_epoch_history(train_history, y_label = "Arbitrary Units", title = 'EpochHistory.png')
    test_loss, test_mae, test_preds_probs, test_preds = test_model(model, test_x, test_y)
    data = {
        'test_loss':test_loss,
        'test_mae':test_mae,
        'test_predictions':test_preds,
        'test_actual':test_y
    }
    for i in range(len(test_preds_probs[0])):
        data['test_probability_%s' % str(i+1)] = [x[i] for x in test_preds_probs]
    return pd.DataFrame(data=data), test_preds_probs

def determine_best_epoch(model,X, Y):
    log = logging.getLogger('determine_best_epoch')
    log.info('Calculating history...')
    model, history = train_model(model, X, Y)
    data = {
        'epoch': history.epoch,
        'train_mae_loss':history.history['mean_absolute_error'],
        'val_mae_loss':history.history['val_mean_absolute_error']
    }
    log.info('Saving...')
    Ann_utilities.save_data('1d25i.csv',pd.DataFrame(data=data))
    log.info('Plotting...')
    return pd.DataFrame(data), history

def determine_best_architecture(X, Y):
    log = logging.getLogger('determine_best_architecture')
    log.info('determine_best_architecture')
    final_df = []
    static_architecture = [112,56,28,14]
    for index in range(len(static_architecture)):
        for depth in range(5):
            if depth == 0:
                model = Ann_utilities.build_shallow_model(X.shape[1])
                model, history = train_model(model,X,Y)
                data = {
                    'Architecture': '0',
                    'best_train_mae_loss':history.history['mean_absolute_error'][-1],
                    'best_val_mae_loss':history.history['val_mean_absolute_error'][-1]
                }
                final_df.append(pd.DataFrame(data=data, index=[0]))
                continue
            hidden_layer_architecture = []
            max = static_architecture[index]
            for i in range(depth):
                hidden_layer_architecture.append(max)
                max = max/2
            log.info('Running with -> %s' % '->'.join(str(x) for x in hidden_layer_architecture))
            model = Ann_utilities.build_abstract_model(X.shape[1], hidden_layer_architecture)
            model, history = train_model(model,X,Y)
            data = {
                'Architecture': '->'.join(str(x) for x in hidden_layer_architecture),
                'best_train_mae_loss':history.history['mean_absolute_error'][-1],
                'best_val_mae_loss':history.history['val_mean_absolute_error'][-1]
            }
            final_df.append(pd.DataFrame(data=data, index=[0]))
    return pd.concat(final_df)

def main():
    pass

if __name__ == "__main__":
    logging.basicConfig(filename='logger.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    ########################
    # Run user-defined model
    sys.exit(main())