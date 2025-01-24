from models.Baysian_regression import BayesianRidgeModel
from models.Linear_regression import LinearRegressionModel
from models.Decision_tree_regression import DecisionTreeRegressionModel
from data.data_setup import merge_data
from data.data_setup import split_data
from data.data_setup import preprocess_data
from metrics.Mean_squared_error import MeanSquaredError
from metrics.Root_mean_squared_error import RootMeanSquaredError
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from pandas import DataFrame


def train(X_train, y_train, X_valid, y_valid) -> tuple:
    """
    This function trains the models on the training data and evaluates
    them on the test data.
    """
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
    print("R2 Score for Linear Regression: "
          f"{linear_regression_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Bayesian Ridge: "
          f"{bayesian_ridge_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Decision Tree Regression: "
          f"{decision_tree_regression_model.model.score(X_valid, y_valid)}")

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

    print("Mean Squared Error for Linear Regression: ")
    print(f"{mse_linear_tuned }")
    print("Mean Squared Error for Bayesian Ridge: ")
    print(f"{mse_bayesian_tuned}")
    print("Mean Squared Error for Decision Tree Regression: ")
    print(f"{mse_decision_tuned}")
    print("Root Mean Squared Error for Linear Regression: ")
    print(f"{rmse_linear_tuned}")
    print("Root Mean Squared Error for Bayesian Ridge: ")
    print(f"{rmse_bayesian_tuned}")
    print("Root Mean Squared Error for Decision Tree Regression: ")
    print(f"{rmse_decision_tuned}")
    print("R2 Score for Linear Regression: ")
    print(f"{linear_regression_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Bayesian Ridge: ")
    print(f"{bayesian_ridge_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Decision Tree Regression: ")
    print(f"{decision_tree_regression_model.model.score(X_valid, y_valid)}")

    # plot the results

    figure, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].scatter(y_valid, pred_linear_tuned, color='blue')
    axes[0, 0].set_title("Linear Regression")
    axes[0, 0].set_xlabel("True Values")
    axes[0, 0].set_ylabel("Predictions")

    axes[0, 1].scatter(y_valid, pred_bayesian_tuned, color='red')
    axes[0, 1].set_title("Bayesian Ridge")
    axes[0, 1].set_xlabel("True Values")
    axes[0, 1].set_ylabel("Predictions")

    axes[1, 0].scatter(y_valid, pred_decision_tuned, color='green')
    axes[1, 0].set_title("Decision Tree Regression")
    axes[1, 0].set_xlabel("True Values")
    axes[1, 0].set_ylabel("Predictions")

    axes[1, 1].scatter(y_valid, pred_linear_tuned, color='blue',
                       label='Linear')
    axes[1, 1].scatter(y_valid, pred_bayesian_tuned, color='red',
                       label='Bayesian')
    axes[1, 1].scatter(y_valid, pred_decision_tuned, color='green',
                       label='Decision Tree')
    axes[1, 1].set_title("Comparison")
    axes[1, 1].set_xlabel("True Values")
    axes[1, 1].set_ylabel("Predictions")
    axes[1, 1].legend()

    plt.show()

    return linear_grid_search.best_params_, bayesian_grid_search.best_params_, dt_grid_search.best_params_


def evaluate(best_params: tuple,
             X_test: DataFrame,
             y_test: DataFrame) -> tuple:
    """
    This function evaluates the models on the test data.

    Parameters
    ----------
    best_params : tuple
        The best parameters for the models.

    Returns
    -------
    mse : tuple
        The Mean Squared Error for the models.
    rmse : tuple
        The Root Mean Squared Error for the models.
    """
    # preprocess the data
    df = merge_data()

    df = preprocess_data(df)
    # split the data
    train_df, test_df, valid_df = split_data(df)

    # get the features and target
    X_train = train_df.drop(columns=["price"]).values
    y_train = train_df["price"].values

    X_test = test_df.drop(columns=["price"]).values
    y_test = test_df["price"].values

    # train the models
    linear_regression_model = LinearRegressionModel()
    linear_regression_model.model = linear_regression_model.model.set_params(
        **best_params[0])
    linear_regression_model.train(X_train, y_train)

    bayesian_ridge_model = BayesianRidgeModel()
    bayesian_ridge_model.model = bayesian_ridge_model.model.set_params(
        **best_params[1])
    bayesian_ridge_model.train(X_train, y_train)

    decision_tree_regression_model = DecisionTreeRegressionModel()
    decision_tree_regression_model.model = decision_tree_regression_model.model.set_params(
        **best_params[2])
    decision_tree_regression_model.train(X_train, y_train)

    # predict the target values
    pred_linear = linear_regression_model.predict(X_test)
    pred_bayesian = bayesian_ridge_model.predict(X_test)
    pred_decision = decision_tree_regression_model.predict(X_test)

    # calculate the metrics
    mse = MeanSquaredError()
    rmse = RootMeanSquaredError()

    mse_linear = mse.calculate(y_test, pred_linear)
    mse_bayesian = mse.calculate(y_test, pred_bayesian)
    mse_decision = mse.calculate(y_test, pred_decision)

    rmse_linear = rmse.calculate(y_test, pred_linear)
    rmse_bayesian = rmse.calculate(y_test, pred_bayesian)
    rmse_decision = rmse.calculate(y_test, pred_decision)

    print("Mean Squared Error for Linear Regression: ")
    print(f"{mse_linear}")
    print("Mean Squared Error for Bayesian Ridge: ")
    print(f"{mse_bayesian}")
    print("Mean Squared Error for Decision Tree Regression: ")
    print(f"{mse_decision}")
    print("Root Mean Squared Error for Linear Regression: ")
    print(f"{rmse_linear}")
    print("Root Mean Squared Error for Bayesian Ridge: ")
    print(f"{rmse_bayesian}")
    print("Root Mean Squared Error for Decision Tree Regression: ")
    print(f"{rmse_decision}")
    print("R2 Score for Linear Regression: ")
    print(f"{linear_regression_model.model.score(X_test, y_test)}")
    print("R2 Score for Bayesian Ridge: ")
    print(f"{bayesian_ridge_model.model.score(X_test, y_test)}")
    print("R2 Score for Decision Tree Regression: ")
    print(f"{decision_tree_regression_model.model.score(X_test, y_test)}")


if __name__ == "__main__":
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

    X_test = test_df.drop(columns=["price"]).values
    y_test = test_df["price"].values

    # train the models
    print("##############################################################")
    print("Training the models:")
    print("##############################################################")
    best_params = train(X_train, y_train, X_valid, y_valid)

    # evaluate the models
    print("##############################################################")
    print("Evaluation on test data:")
    print("##############################################################")
    evaluate(best_params, X_test, y_test)

    print("##############################################################")
    print("Done!")
    print("##############################################################")
