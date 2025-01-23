from models.Baysian_regression import BayesianRidgeModel
from models.Linear_regression import LinearRegressionModel
from models.Decision_tree_regression import DecisionTreeRegressionModel
from data.data_setup import merge_data
from data.data_setup import split_data
from data.data_setup import preprocess_data
from metrics.Mean_squared_error import MeanSquaredError
from metrics.Root_mean_squared_error import RootMeanSquaredError
from sklearn.model_selection import GridSearchCV


def train() -> None:
    """
    This function trains the models on the training data and evaluates
    them on the test data.
    """
    # preprocess the data
    df = merge_data()

    df = preprocess_data(df)
    # split the data
    train_df, test_df, valid_df = split_data(df)

    # get the features and target
    X_train = train_df.drop(columns=["price"]).values
    y_train = train_df["price"].values

    X_valid = valid_df.drop(columns=["price"]).values
    y_valid = valid_df["price"].values

    # train the models
    linear_regression_model = LinearRegressionModel()
    linear_regression_model.train(X_train, y_train)

    bayesian_ridge_model = BayesianRidgeModel()
    bayesian_ridge_model.train(X_train, y_train)

    decision_tree_regression_model = DecisionTreeRegressionModel()
    decision_tree_regression_model.train(X_train, y_train)

    # predict the target values
    pred_linear = linear_regression_model.predict(X_valid)
    pred_bayesian = bayesian_ridge_model.predict(X_valid)
    pred_decision = decision_tree_regression_model.predict(X_valid)

    # calculate the metrics
    mse = MeanSquaredError()
    rmse = RootMeanSquaredError()

    mse_linear = mse.calculate(y_valid, pred_linear)
    mse_bayesian = mse.calculate(y_valid, pred_bayesian)
    mse_decision = mse.calculate(y_valid, pred_decision)

    rmse_linear = rmse.calculate(y_valid, pred_linear)
    rmse_bayesian = rmse.calculate(y_valid, pred_bayesian)
    rmse_decision = rmse.calculate(y_valid, pred_decision)

    print(f"Mean Squared Error for Linear Regression: "
          f"{mse_linear}")
    print(f"Mean Squared Error for Bayesian Ridge: "
          f"{mse_bayesian}")
    print(f"Mean Squared Error for Decision Tree Regression: "
          f"{mse_decision}")

    print(f"Root Mean Squared Error for Linear Regression: "
          f"{rmse_linear}")
    print(f"Root Mean Squared Error for Bayesian Ridge: "
          f"{rmse_bayesian}")
    print("Root Mean Squared Error for Decision Tree Regression: "
          f"{rmse_decision}")

    # alter parameters with validate split
    # Define hyperparameter grids
    linear_param_grid = {
        'fit_intercept': [True, False]
    }
    bayesian_param_grid = {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    }

    decision_tree_param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Linear Regression Hyperparameter Tuning
    linear_grid_search = GridSearchCV(estimator=linear_regression_model.model,
                                      param_grid=linear_param_grid, 
                                      cv=5, scoring='neg_mean_squared_error',
                                      verbose=2)
    linear_grid_search.fit(X_train, y_train)
    print("Best Parameters for Linear Regression:", linear_grid_search.
          best_params_)

    # Bayesian Ridge Hyperparameter Tuning
    bayesian_grid_search = GridSearchCV(estimator=bayesian_ridge_model.model,
                                        param_grid=bayesian_param_grid, cv=5,
                                        scoring='neg_mean_squared_error',
                                        verbose=2)
    bayesian_grid_search.fit(X_train, y_train)
    print("Best Parameters for Bayesian Ridge:", bayesian_grid_search.
          best_params_)

    # Decision Tree Hyperparameter Tuning
    dt_grid_search = GridSearchCV(estimator=decision_tree_regression_model.
                                  model,
                                  param_grid=decision_tree_param_grid,
                                  cv=5, scoring='neg_mean_squared_error',
                                  verbose=2)
    dt_grid_search.fit(X_train, y_train)
    print("Best Parameters for Decision Tree Regression:",
          dt_grid_search.best_params_)

    # Retrain models with best parameters
    linear_regression_model.model = linear_grid_search.best_estimator_
    bayesian_ridge_model.model = bayesian_grid_search.best_estimator_
    decision_tree_regression_model.model = dt_grid_search.best_estimator_

    linear_regression_model.train(X_train, y_train)
    bayesian_ridge_model.train(X_train, y_train)
    decision_tree_regression_model.train(X_train, y_train)

    # Re-evaluate models on validation set
    pred_linear_tuned = linear_regression_model.predict(X_valid)
    pred_bayesian_tuned = bayesian_ridge_model.predict(X_valid)
    pred_decision_tuned = decision_tree_regression_model.predict(X_valid)

    mse_linear_tuned = mse.calculate(y_valid, pred_linear_tuned)
    mse_bayesian_tuned = mse.calculate(y_valid, pred_bayesian_tuned)
    mse_decision_tuned = mse.calculate(y_valid, pred_decision_tuned)

    rmse_linear_tuned = rmse.calculate(y_valid, pred_linear_tuned)
    rmse_bayesian_tuned = rmse.calculate(y_valid, pred_bayesian_tuned)
    rmse_decision_tuned = rmse.calculate(y_valid, pred_decision_tuned)

    print(f"Tuned Mean Squared Error for Linear Regression: "
          f"{mse_linear_tuned}")
    print(f"Tuned Mean Squared Error for Bayesian Ridge: "
          f"{mse_bayesian_tuned}")
    print(f"Tuned Mean Squared Error for Decision Tree Regression: "
          f"{mse_decision_tuned}")
    print(f"Tuned Root Mean Squared Error for Linear Regression: "
          f"{rmse_linear_tuned}")
    print(f"Tuned Root Mean Squared Error for Bayesian Ridge: "
          f"{rmse_bayesian_tuned}")
    print(f"Tuned Root Mean Squared Error for Decision Tree Regression: "
          f"{rmse_decision_tuned}")


if __name__ == "__main__":
    train()
