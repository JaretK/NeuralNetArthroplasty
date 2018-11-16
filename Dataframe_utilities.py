"""
Utilities for use in manipulating pandas dataframes
"""

import logging
import pandas as pd
import sys
from six import string_types

class DataframeProcessor:
    raw_df = 0
    def __init__(self, dataframe):
        self.raw_df = dataframe
    def labelEncode(self, consume_raw_df = True):
        from sklearn.preprocessing import LabelEncoder
        fit = self.raw_df.apply(LabelEncoder().fit_transform)
        if consume_raw_df:
            self.raw_df = 0 # free memory
        return fit
    def make_dummies(self, columns, consume_raw_df = True):
        fit = pd.get_dummies(self.raw_df, columns = columns)
        if consume_raw_df:
            self.raw_df = 0 # free memory
        return fit
    @staticmethod
    def normalize_dummy_columns(train, test):
        """
        Adds dummy columns to the train set that are found in the test set
        """
        # Get missing columns in the test test
        test_missing_cols = set( train.columns ) - set( test.columns )
        train_missing_cols = set( test.columns ) - set( train.columns )
        # Add a missing column in test set with default value equal to 0
        for c in test_missing_cols:
            test[c] = 0
        for c in train_missing_cols:
            train[c] = 0
        # Ensure the order of columns is the same in both
        test = test[train.columns]
        train = train[train.columns]
        return train, test

    @staticmethod
    def dropna_inplace(df, column):
        if type(column) is not list and _string_type(column):
            column = [column]
        before = df.shape[0]
        df.dropna(subset = column, inplace=True)
        after = df.shape[0]
        log = logging.getLogger('pandas.dropna')
        log.info('Dropna delta = %d' % (before-after))
    @staticmethod
    def dropna_all_inplace(df):
        before = df.shape[0]
        df.dropna(how='all', inplace=True)
        after = df.shape[0]
        log = logging.getLogger('pandas.dropna')
        log.info('Dropna delta = %d' % (before-after))


def load_data(data_file, verbose = False):
    """
    Params:
    data_file: .csv containing the data to model.

    Return:
    a pandas dataframe of the data_file
    """
    log = logging.getLogger('load.data')
    log.info('Loading %s' % data_file)
    df = pd.read_csv(data_file, low_memory=False)
    log.info('Loaded X: %s obs of %s dimensions' % df.shape)
    if verbose:
        sys.stdout.write('{}\n'.format(df.info()))
    return df

def get_x(dataframe, attrs):
    '''
    Returns a dataframe that keeps only the categorical encoded attrs
    '''
    x = keep_columns(
                dataframe, attrs.get_all_attributes())
    x = DataframeProcessor(x).make_dummies(columns = attrs.categorical_attributes)
    x = normalize_df(x, attrs.normalized_attributes)
    return x

def _string_type(obj):
    """
    Helper function that returns true iff obj contains only string elements
    obj = 'this is a string' -> True
    obj = [1,2,3,'string1', 'string2'] -> False
    obj = ['string1', 'string2', '3'] -> True
    """
    try:
        return bool(obj) and all(isinstance(elem, string_types) for elem in obj)
    except:
        return False


def keep_columns(df, *args):
    """
    Keep columns that appear in args.

    Params:
    *args: strings or list of strings to keep in the dataframe

    Return: pandas dataframe with only the elements that appear in *args
    """
    to_keep = []
    for arg in args:
        if _string_type(arg):
            to_keep.extend(arg)
        else:
            raise TypeError('%s must be a string type container' % arg)
    return df[to_keep]

def get_mean_std_df(df, column_label):
    """
    @Params:
    df = pandas dataframe on which to operate
    column_label = column to return mean and std

    @Return:
    mean, std for column_label
    """
    mean = df[column_label].mean(axis=0)
    std = df[column_label].std(axis=0)
    return mean, std

def get_mean_std_series(series):
    """
    @Params:
    series = pandas series object

    @Return:
    mean, std for series
    """
    from pandas import Series as pdSeries
    if not isinstance(series, pdSeries):
        raise TypeError('Input must be pandas.Series, was %s' % type(series))
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    return mean, std

def normalize_series(series, mean_x = None, std_x = None):
    """
    Converts @param series object into Z-scores
    @Return:
    pandas.Series object of Z-scores
    """
    from pandas import Series as pdSeries
    if not isinstance(series, pdSeries):
        raise TypeError('Input must be pandas.Series, was %s' % type(series))
    mean, std = get_mean_std_series(series)
    # overwrite mean / std with given inputs
    if mean_x != None:
        mean = mean_x
    if std_x != None:
        std = std_x
    s = (series-mean)/std
    return s


def normalize_df(df, normalize_labels_list, mean_x = None, std_x = None):
    """
    Normalize columns in normalize_labels_list into Z-scores

    @Params:
    df = pandas dataframe on which to operate
    normalize_labels_list = list of column labels to convert to Z-scores
    """
    if normalize_labels_list == None or len(normalize_labels_list) == 0:
        return df
    for x in normalize_labels_list:
        # normalize_labels_list must be a string type (unicode, string, etc)
        if not isinstance(x, string_types):
            raise TypeError('%s type must inherit basestring')
        mean, std = get_mean_std_df(df, x)
        if mean_x != None:
            mean = mean_x
        if std_x != None:
            std = std_x
        df[x] = (df[x] - mean)/std
    return df


def get_dummy_dataset():
    from pandas import DataFrame
    """
    Returns a dummy dataset for small test cases
    Mainly used for testing utility functions
    """
    return DataFrame({'A': [1, 2, 3, 4],
                      'B': ["Yes", "No", "Yes", "Yes"],
                      'C': ["Yes", "No", "No", "Yes"],
                      'D': ["No", "Yes", "No", "Yes"],
                      'E': ["Maybe", "Yes", "No", "Yes"]})

def get_dummy_series():
    # import pandas, keep function separate from module
    from pandas import Series
    """
    Returns a dummy series for small test cases
    Mainly used for testing utility functions
    """
    data = {'a' : 0., 'b' : 1., 'c' : 2.}
    return Series(data)

if __name__ == '__main__':
    # test DataframeProcessor.normalize_dummy_columns
    train = pd.DataFrame(data={
    'A_1':[1,2,3,4],
    'A_2':[5,6,7,8],
    'B_1':['a','b','a','c'],
    })
    test = pd.DataFrame(data={
    'A_3':[5,6,7,8],
    'B_2':['a','b','a','c'],
    'A_1':[1,2,3,4]
    })
    train, test = DataframeProcessor.normalize_dummy_columns(train, test)
    print(train)
    print(test)
