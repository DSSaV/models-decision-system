
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

from tcn import TCN, tcn_full_summary


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
    # denormalized_predictions = scaler.inverse_transform(predictions)

    return {
        'model': model,
        'predictions': predictions
    }


def add_lstm_layer(model, data, index, name, settings):
    """Support function used to add a Keras Layers to the LSTM model."""

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


def add_lstm_layers(model, data, settings):
    """Support function that loops through all available Keras Layers."""
    # LOOP THROUGH REQUESTED MODEL LAYERS
    for index, layer in enumerate(settings['layers']):
        # LAYER PROPS
        name = list(layer)[0]
        params = layer[name]

        # GENERATE & ADD THE LAYER
        add_lstm_layer(model, data, index, name, params)


def long_short_term_memory(data, settings):
    """Creates a Long short-term memory model (LSTM) and predictions.

    Args:
        data: pandas.DataFrame.
        settings: Dictionary object containing settings parameters.
    Returns:
        A dictionary containing the LSTM model and predictions.
    """

    #  TODO: FIX LABEL LEAKAGE
    #  VARIABLES
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_validation = data['validation']['features']
    y_validation = data['validation']['labels']
    x_test = data['test']['features']
    scaler = data['scaler']

    #  INSTANTIATE MODEL
    model = Sequential()

    #  TODO:TRAIN DATA GENERATOR
    #  CODE COMES HERE

    #  ADDING LAYERS TO MODEL
    add_lstm_layers(model, data, settings)

    #  COMPILE THE MODEL
    model.compile(
        loss=settings['loss'],
        optimizer=settings['optimizer']
    )

    #  TODO: change fit() --> fit_generator()
    #  TRAIN USING TRAIN DATA
    model.fit(
        x_train,
        y_train,
        epochs=settings['epochs'],
        batch_size=settings['batch'],

        #  TODO: REMOVE VALIDATION
        #  ADD VALIDATION DATA
        validation_data=(
            x_validation,
            y_validation
        ),

        # VALIDATE EVERY 25 STEPS
        validation_steps=settings['validation']
    )

    #  TODO:TEST DATA GENERATOR

    #  TODO:update to use test generator
    #  PREDICT USING TEST DATA
    predictions = model.predict(x_test)

    # denormalized_predictions = ""

    return {
        'model': model,
        'predictions': predictions
    }


def add_base_layer_to_tcn(name, model_output, settings, index):
    """Support function that adds Keras Layers to TCN model."""
    # AVAILABLE LAYERS
    available = {
        'dropout': Dropout,
        'dense': Dense
    }

    # SELECT THE CORRECT FUNCTION
    func = available[name]

    if name == 'dropout':
        return func(settings['value'])(model_output)

    else:
        return func(settings['value'])(model_output)


def add_tcn_layers(model_input, settings):
    """Support function that adds TCN Layer and requested Keras Layers to the TCN model"""

    # STARTING LAYER (TCN)
    tcn_string = ''
    try:
        for index, stats in enumerate(settings['layers'][0]['tcn']):
            # LAYER PROPS
            value = settings['layers'][0]['tcn'][stats]
            if list(settings['layers'][0]['tcn'])[-1]:
                tcn_string += str(stats) + '=' + str(value)
            else:
                tcn_string += str(stats) + '=' + str(value) + ','
        model_output = TCN(tcn_string)(model_input)

    except ValueError:
        model_output = TCN(return_sequences=False)(model_input)

    for index, layer in enumerate(settings['layers']):

        # LAYER PROPS
        name = list(layer)[0]
        params = layer[name]

        if index == 0:
            continue
        else:
            model_output = add_base_layer_to_tcn(name, model_output, params, index)

    return model_output


def temporal_convolutional_network(data, settings):
    """Creates a Temporal Convolutional Network model (TCN) and predictions.

        Args:
            data: pandas.DataFrame.
            settings: Dictionary object containing settings parameters.
        Returns:
            A dictionary containing the TCN model and predictions.
        """

    #  VARIABLES
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_validation = data['validation']['features']
    y_validation = data['validation']['labels']
    x_test = data['test']['features']
    scaler = data['scaler']
    timesteps = x_train.shape[0]
    input_dim_x = x_train.shape[1]
    input_dim_y = x_train.shape[2]
    batch_size = settings['batch']

    #  TODO: ADD GENERATOR

    #  INSTANTIATE KERAS TENSOR WITH Input()
    model_input = Input(shape=(input_dim_x, input_dim_y))

    #  INSTANTIATE MODEL LAYERS
    model_output = add_tcn_layers(model_input, settings)

    #  INSTANTIATE MODEL AND ASSIGN INPUT AND OUTPUT
    model = Model(inputs=[model_input], outputs=[model_output])

    # COMPILE THE MODEL
    model.compile(optimizer=settings['optimizer'], loss=settings['loss'])

    #  PRINT MODEL STATS
    tcn_full_summary(model, expand_residual_blocks=False)

    #  TODO: CHNAGE TO fit_generator()
    #  TRAIN THE MODEL WITH VALIDATION
    model.fit(x_train, y_train, epochs=settings['epochs'], validation_data=(x_validation, y_validation))

    # PREDICT USING TEST DATA
    predictions = model.predict(x_test)

    return {
        'model': model,
        'predictions': predictions
    }


def linear_support_vector_classifier(data, settings):
    """Creates a Linear Support Vector Classifier (Linear SVC) and predictions.

    Args:
        data: pandas.DataFrame.
        settings: Dictionary object containing settings parameters, Labels have to be buy/sell/hold or another classification type
    Returns:
        A dictionary containing the Linear SVC model and predictions.
    """

    #  VARIABLES
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_test = data['test']['features']
    scaler = data['scaler']

    # INSTANTIATE MODEL
    model = LinearSVC()

    # CREATE PREDICTIONS USING TRAIN DATA
    model.fit(x_train, y_train)

    # PREDICTIONS
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
        'tcn': temporal_convolutional_network,
        'svc': linear_support_vector_classifier
    }

    # SELECT THE CORRECT FUNCTION & START
    return model[name](dataset, settings)
