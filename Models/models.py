from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

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

    #  INSTANTIATE MODEL
    linear = LinearRegression()

    #  CREATE PREDICTIONS USING TEST DATA
    model = linear.fit(x_train, y_train)

    #  PREDICTIONS
    predictions = model.predict(x_test)

    return {
        'model': model,
        'predictions': predictions
    }


def create_generator(dataset, params, shuffle=True):
    # DECONSTRUCT DATASET
    features = dataset['features']
    labels = dataset['labels']

    # DECONSTRUCT PARAMS
    batch = params['batch']
    window = params['window']

    # GENERATE & RETURN
    return TimeseriesGenerator(
        features,
        labels,
        length=window,
        batch_size=batch,
        shuffle=shuffle
    )


def add_lstm_layer(model, data, index, name, settings, shape):
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
    model.add(func(**settings))


def add_lstm_layers(model, data, settings, shape):
    """Support function that loops through all available Keras Layers."""

    # LOOP THROUGH REQUESTED MODEL LAYERS
    for index, layer in enumerate(settings['layers']):
        # LAYER PROPS
        name = list(layer)[0]
        params = layer[name]

        # GENERATE & ADD THE LAYER
        add_lstm_layer(model, data, index, name, params, shape)


def long_short_term_memory(data, settings):
    """Creates a Long short-term memory model (LSTM) and predictions.

    Args:
        data: pandas.DataFrame.
        settings: Dictionary object containing settings parameters.
    Returns:
        A dictionary containing the LSTM model and predictions.
    """

    #  INSTANTIATE MODEL
    model = Sequential()

    #  TRAIN DATA GENERATOR
    train_generator = create_generator(
        data['train'],
        settings['morph'],
        shuffle=True
    )

    #  ADDING LAYERS TO MODEL
    add_lstm_layers(model, data, settings, train_generator[0][0].shape)

    #  COMPILE THE MODEL
    model.compile(
        loss=settings['loss'],
        optimizer=settings['optimizer']
    )

    #  TRAIN USING TRAIN DATA
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=settings['epochs'],
        verbose=0
    )

    #  TEST DATA GENERATOR
    test_generator = create_generator(
        data['test'],
        settings['morph'],
        shuffle=False
    )

    #  PREDICT USING TEST DATA
    predictions = model.predict(test_generator)

    # denormalized_predictions = ""

    return {
        'model': model,
        'predictions': predictions
    }


def add_tcn_layer(name, model_output, settings, index):
    """Support function that adds Keras Layers to TCN model."""

    # AVAILABLE LAYERS
    available = {
        'dropout': Dropout,
        'dense': Dense
    }

    # SELECT THE CORRECT FUNCTION
    func = available[name]
    return func(**settings)(model_output)


def add_tcn_layers(model_input, settings):
    """Support function that adds TCN Layer and requested Keras Layers to the TCN model"""

    layers = settings['layers']

    try:
        model_output = TCN(**settings['layers'][0]['tcn'])(model_input)
    except ValueError as e:
        print(e, 'Wrong structure on yaml config file')

    for index, layer in enumerate(layers):
        name = list(layer)[0]
        params = layer[name]
        if name == 'tcn':
            continue
        else:
            model_output = add_tcn_layer(name, model_output, params, index)

    return model_output


def temporal_convolutional_network(data, settings):
    """Creates a Temporal Convolutional Network model (TCN) and predictions.

        Args:
            data: pandas.DataFrame.
            settings: Dictionary object containing settings parameters.
        Returns:
            A dictionary containing the TCN model and predictions.
        """

    #  TRAIN DATA GENERATOR
    train_generator = create_generator(
        data['train'],
        settings['morph'],
        shuffle=True
    )
    #  TRAIN DATA GENERATOR
    test_generator = create_generator(
        data['test'],
        settings['morph'],
        shuffle=False
    )

    #  INSTANTIATE KERAS TENSOR INPUT WITH TIMESERIESGENEREATOR SHAPE
    model_input = Input(batch_shape=train_generator[0][0].shape)

    #  INSTANTIATE MODEL LAYERS
    model_output = add_tcn_layers(model_input, settings)

    #  INSTANTIATE MODEL AND ASSIGN INPUT AND OUTPUT
    model = Model(inputs=[model_input], outputs=[model_output])

    # COMPILE THE MODEL
    model.compile(optimizer=settings['optimizer'], loss=settings['loss'])

    #  PRINT MODEL STATS
    tcn_full_summary(model, expand_residual_blocks=False)

    #  TRAIN THE MODEL WITH VALIDATION
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=settings['epochs'],
        verbose=0
    )

    #  PREDICT USING TEST DATA
    predictions = model.predict(test_generator)

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

def grid_search_model(model, data, settings):
    """Runs a grid search on the given model to define the best parameters for the model. 
    
    Parameters:
    -----------
    model: sklearn model 
    unfitted model
    dataset: dict
    Dictionary that contains x_train, y_train, features and labels
    
    Returns:
    --------
    model: sklearn model
    Same type of model as the input model, with optimized parameters
    """
    #  VARIABLES
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_test = data['test']['features']
    y_test = data['test']['labels']
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    
    # CREATE GRID SEARCH CV
    grid_search = GridSearchCV(estimator = model, param_grid = settings["grid_params"], 
                          cv = settings["cv"], n_jobs = settings["n_jobs"], verbose = 2)
    
    # FIT GRIDSEARCH
    grid_search.fit(x, y)
    
    # PICK OUT THE BEST MODEL FROM GRID SEARCH
    model = grid_search.best_estimator_
    
    # RUN MODEL ON BEST PARAMS   
    model.fit(x_train, y_train)
    
    # RETURN MODEL WITH BEST PARAMS    
    return model


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
