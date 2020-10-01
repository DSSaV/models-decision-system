import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.python.keras import Sequential, Input, Model
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

    # TODO: fix denormalization
    #  DENORMALIZED PREDICTIONS FOR ACTUAL PREDICTION VALUES
    #denormalized_predictions = scaler.inverse_transform(predictions)

    return {
        'model': model,
        'predictions': predictions
    }


def add_layer(model, data, index, name, settings):
    # AVAILABLE LAYERS
    available = {
        'lstm': LSTM,
        'dropout': Dropout,
        'dense': Dense
    }

    # SELECT THE CORRECT FUNCTION
    func = available[name]

    # IF AN ACTIVATION IS FOUND & THIS IS THE FIRST LAYER
    if 'activation' in settings and index == 0:
        model.add(func(
            settings['value'],
            activation=settings['activation'],
            input_shape=(data['train']['features'].shape[1], 1)
        ))

    # JUST AN ACTIVATION FUNCTION
    elif 'activation' in settings:
        model.add(func(
            settings['value'],
            activation=settings['activation']
        ))

    # OTHERWISE, DEFAULT TO JUST USING THE VALUE
    else:
        model.add(func(
            settings['value']
        ))


def add_layers(model, data, settings):
    # LOOP THROUGH REQUESTED MODEL LAYERS
    for index, layer in enumerate(settings['layers']):
        # LAYER PROPS
        name = list(layer)[0]
        params = layer[name]

        # GENERATE & ADD THE LAYER
        add_layer(model, data, index, name, params)


def long_short_term_memory(data, settings):
    """Creates a Long short-term memory model (LSTM) and predictions.

    Args:
        data: pandas.DataFrame.
        settings: Dictionary object containing settings parameters.
    Returns:
        A dictionary containing the LSTM model and predictions.
    """

    #  VARIABLES
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_validation = data['validation']['features']
    y_validation = data['validation']['labels']
    x_test = data['test']['features']
    scaler = data['scaler']

    #  INSTANTIATE MODEL
    model = Sequential()

    #  ADDING LAYERS TO MODEL
    add_layers(model, data, settings)

    # COMPILE THE MODEL
    model.compile(
        loss=settings['loss'],
        optimizer=settings['optimizer']
    )

    # TRAIN USING TRAIN DATA
    model.fit(
        x_train,
        y_train,
        epochs=settings['epochs'],
        batch_size=settings['batch'],

        # VALIDATION_SPLIT WORKS NORMALLY, BUT SUCKS FOR TIMESERIES
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        # goal https://miro.medium.com/max/300/1*R09z6KNJuMkSmRcPApj9cQ.png

        # ADD VALIDATION DATA
        validation_data=(
            x_validation,
            y_validation
        ),

        # VALIDATE EVERY 25 STEPS
        validation_steps=settings['validation']
    )

    # PREDICT USING TEST DATA
    predictions = model.predict(x_test)

    #denormalized_predictions = ""

    return {
        'model': model,
        'predictions': predictions
    }


def temporal_convolutional_network(data, settings):

    #  VARIABLES
    batch_size = settings['batch']
    timesteps = 14
    input_dim = len(data['train']['features'])
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_validation = data['validation']['features']
    y_validation = data['validation']['labels']
    x_test = data['test']['features']
    scaler = data['scaler']

    model_input = Input(batch_shape=(batch_size, timesteps, input_dim))

    model_output = TCN(return_sequences=False)(model_input)  # The TCN layers are here.
    model_output = Dropout(0.15)(model_output)
    model_output = Dense(50)(model_output)
    model_output = Dense(1)(model_output)

    model = Model(inputs=[model_input], outputs=[model_output])
    model.compile(optimizer='adam', loss='mse')

    model.fit(x_train, y_train, epochs=settings['epochs'], validation_data=(x_validation, y_validation))
    #  https://keras.io/api/models/model_training_apis/#fit-method

    predictions = model.predict(x_test)

    return {
        'model': model,
        'predictions': predictions
    }


def train_model(dataset, name, settings):
    # AVAILABLE MODELS
    model = {
        'linreg': linear_regression,
        'lstm': long_short_term_memory,
        'tcn': temporal_convolutional_network
    }

    # SELECT THE CORRECT FUNCTION & START
    return model[name](dataset, settings)
