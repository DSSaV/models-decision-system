import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tcn import TCN


def linear_regression(data, settings):
    """Creates a linear regression model and predictions.

    Args:
        data: pandas.DataFrame.
        settings: Dictionary object containing settings parameters.
    Returns:
        A dictionary containing the linear regression model and predictions.
    """

    #  VARIABLES
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_test = data['test']['features']
    scaler = data['scaler']

    #  INSTANTIATE MODEL
    linear = LinearRegression()

    #  CREATE PREDICTIONS USING TEST DATA
    model = linear.fit(x_train, y_train)

    #  PREDICTIONS
    predictions = model.predict(x_test)

    #  DENORMALIZED PREDICTIONS FOR ACTUAL PREDICTION VALUES
    denormalized_predictions = scaler.inverse_transform(predictions)

    return {
        'model': model,
        'predictions': denormalized_predictions
    }


def long_short_term_memory(data, settings):
    """Creates a Long short-term memory model (LSTM) and predictions.

    Args:
        data: pandas.DataFrame.
        settings: Dictionary object containing settings parameters.
    Returns:
        A dictionary containing the LSTM model and predictions.
    """

    #  VARIABLES

    #  INSTANTIATE MODEL
    model = Sequential()

    denormalized_predictions = ""

    return {
        'model': model,
        'predictions': denormalized_predictions
    }
