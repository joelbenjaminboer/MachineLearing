from models.Baysian_regression import BayesianRidgeModel
from models.Linear_regression import LinearRegressionModel
from models.Decision_tree_regression import DecisionTreeRegressionModel
from models.DNN import build_DNN
from data.data_setup import merge_data
from data.data_setup import split_data
from data.data_setup import preprocess_data
from metrics.Mean_squared_error import MeanSquaredError
from metrics.Root_mean_squared_error import RootMeanSquaredError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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

    Deep_Neural_Network = build_DNN(input_shape=(X_train.shape[1],))
    Deep_Neural_Network.fit(X_train,
                            y_train,
                            validation_split=0.2,
                            epochs=50,
                            batch_size=32)

    # predict the target values
    pred_linear = linear_regression_model.predict(X_valid)
    pred_bayesian = bayesian_ridge_model.predict(X_valid)
    pred_decision = decision_tree_regression_model.predict(X_valid)
    pred_DNN = Deep_Neural_Network.predict(X_valid)

    # calculate the metrics
    mse = MeanSquaredError()
    rmse = RootMeanSquaredError()

    mse_linear = mse.calculate(y_valid, pred_linear)
    mse_bayesian = mse.calculate(y_valid, pred_bayesian)
    mse_decision = mse.calculate(y_valid, pred_decision)
    mse_DNN = mse.calculate(y_valid, pred_DNN)

    rmse_linear = rmse.calculate(y_valid, pred_linear)
    rmse_bayesian = rmse.calculate(y_valid, pred_bayesian)
    rmse_decision = rmse.calculate(y_valid, pred_decision)
    rmse_DNN = rmse.calculate(y_valid, pred_DNN)

    print(f"Mean Squared Error for Linear Regression: "
          f"{mse_linear}")
    print(f"Mean Squared Error for Bayesian Ridge: "
          f"{mse_bayesian}")
    print(f"Mean Squared Error for Decision Tree Regression: "
          f"{mse_decision}")
    print(f"Mean Squared Error for Deep Neural Network: "
          f"{mse_DNN}")
    print("")
    print(f"Root Mean Squared Error for Linear Regression: "
          f"{rmse_linear}")
    print(f"Root Mean Squared Error for Bayesian Ridge: "
          f"{rmse_bayesian}")
    print("Root Mean Squared Error for Decision Tree Regression: "
          f"{rmse_decision}")
    print("Root Mean Squared Error for Deep Neural Network: "
          f"{rmse_DNN}")
    print("")
    print("R2 Score for Linear Regression: "
          f"{linear_regression_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Bayesian Ridge: "
          f"{bayesian_ridge_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Decision Tree Regression: "
          f"{decision_tree_regression_model.model.score(X_valid, y_valid)}")
    print("R2 Score for Deep Neural Network: "
          f"{r2_score(y_valid, pred_DNN)}")

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
    print("")
    print("Root Mean Squared Error for Linear Regression: ")
    print(f"{rmse_linear_tuned}")
    print("Root Mean Squared Error for Bayesian Ridge: ")
    print(f"{rmse_bayesian_tuned}")
    print("Root Mean Squared Error for Decision Tree Regression: ")
    print(f"{rmse_decision_tuned}")
    print("")
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

    return linear_regression_model, bayesian_ridge_model, decision_tree_regression_model, Deep_Neural_Network


def evaluate(best_models: tuple,
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

    # train the models
    linear_regression_model = best_models[0]

    bayesian_ridge_model = best_models[1]

    decision_tree_regression_model = best_models[2]

    Deep_Neural_Network = best_models[3]

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
    print("Mean Squared Error for Deep Neural Network: ")
    print(f"{mse.calculate(y_test, Deep_Neural_Network.predict(X_test))}")
    print("")
    print("Root Mean Squared Error for Linear Regression: ")
    print(f"{rmse_linear}")
    print("Root Mean Squared Error for Bayesian Ridge: ")
    print(f"{rmse_bayesian}")
    print("Root Mean Squared Error for Decision Tree Regression: ")
    print(f"{rmse_decision}")
    print("Root Mean Squared Error for Deep Neural Network: ")
    print(f"{rmse.calculate(y_test, Deep_Neural_Network.predict(X_test))}")
    print("")
    print("R2 Score for Linear Regression: ")
    print(f"{linear_regression_model.model.score(X_test, y_test)}")
    print("R2 Score for Bayesian Ridge: ")
    print(f"{bayesian_ridge_model.model.score(X_test, y_test)}")
    print("R2 Score for Decision Tree Regression: ")
    print(f"{decision_tree_regression_model.model.score(X_test, y_test)}")
    print("R2 Score for Deep Neural Network: ")
    print(f"{r2_score(y_test, Deep_Neural_Network.predict(X_test))}")


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

    # normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # train the models
    print("##############################################################")
    print("Training the models:")
    print("##############################################################")
    best_models = train(X_train, y_train, X_valid, y_valid)

    # evaluate the models
    print("##############################################################")
    print("Evaluation on test data:")
    print("##############################################################")
    evaluate(best_models, X_test, y_test)

    print("##############################################################")
    print("Done!")
    print("##############################################################")
