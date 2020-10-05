# Model & Decisions

This part of the project is composed of creating the models and making a decision based on the output of the models.
It should be able to succeed in the following tasks:

1. Create and update models
2. Create an ensemble
3. Decide on a course of action based on a model or ensemble
4. Analyze the different models

## Models

Out initial plan is to implement the following models.

- Linear Regression
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- LSTM
- [TCNN](https://github.com/philipperemy/keras-tcn)
- Cross Validation SVR
- Log Regression
- Cross Validation SVC
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

This is not a complete list and might change during the span of the copurse.

Each model should be able to:

1. Be updated independently
2. Be used independently without updating
3. Be updated and used in succession?
4. Be used in an ensemble

### Ensemble

This project will include the ability to run models in an ensemble, which means that the models will have their results compared against eachother so that the result isn't affected by a single models outlier value.

The models in the ensemble will be able to be freely chosen between all the available models.

## Decisions

Our plan is to implement a decision system which buys when the value is at the expected lowest, and holds until the values reach the highest point, at which point it then sells. One way we can succeed in this is by looking at the momentum of the forecasted values and compare them against the Close price.

