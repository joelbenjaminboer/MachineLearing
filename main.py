from models.Baysian_regression import BayesianRidgeModel
from models.Linear_regression import LinearRegressionModel
from models.Decision_tree_regression import DecisionTreeRegressionModel
from models.FNN import build_FNN
from data.data_setup import merge_data
from data.data_setup import split_data
from data.data_setup import preprocess_data
from metrics.Mean_squared_error import MeanSquaredError
from metrics.Root_mean_squared_error import RootMeanSquaredError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pandas import DataFrame


def train(X_train: DataFrame,
          y_train: DataFrame,
          X_valid: DataFrame,
          y_valid: DataFrame) -> tuple:
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

    Feedforward_Neural_Network = build_FNN(input_shape=(X_train.shape[1],))
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=20,
                                   restore_best_weights=True
                                   )
    Feedforward_Neural_Network.fit(X_train,
                                   y_train,
                                   validation_split=0.2,
                                   epochs=80,
                                   batch_size=32,
                                   callbacks=[early_stopping],
                                   verbose=1)

    # predict the target values
    pred_linear = linear_regression_model.predict(X_valid)
    pred_bayesian = bayesian_ridge_model.predict(X_valid)
    pred_decision = decision_tree_regression_model.predict(X_valid)
    pred_FNN = Feedforward_Neural_Network.predict(X_valid)

    # print the metrics
    print("Before Hyperparameter Tuning:")
    print_all_metrics(y_valid,
                      X_valid,
                      [pred_linear,
                       pred_bayesian,
                       pred_decision,
                       pred_FNN])

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
    pred_FNN = Feedforward_Neural_Network.predict(X_valid)

    # print the metrics
    print("After Hyperparameter Tuning:")
    print_all_metrics(y_valid,
                      X_valid,
                      [pred_linear_tuned,
                       pred_bayesian_tuned,
                       pred_decision_tuned,
                       pred_FNN])

    # plot the results
    plot_results([pred_linear_tuned,
                  pred_bayesian_tuned,
                  pred_decision_tuned,
                  pred_FNN], y_valid)

    return (linear_regression_model,
            bayesian_ridge_model,
            decision_tree_regression_model,
            Feedforward_Neural_Network)


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

    Feedforward_Neural_Network = best_models[3]

    # predict the target values
    pred_linear = linear_regression_model.predict(X_test)
    pred_bayesian = bayesian_ridge_model.predict(X_test)
    pred_decision = decision_tree_regression_model.predict(X_test)
    pred_FNN = Feedforward_Neural_Network.predict(X_test)

    # calculate the metrics
    print_all_metrics(y_test,
                      X_test,
                      [pred_linear,
                       pred_bayesian,
                       pred_decision,
                       pred_FNN])

    # plot the results
    plot_results([pred_linear,
                  pred_bayesian,
                  pred_decision,
                  pred_FNN], y_test)


def print_all_metrics(y_df: DataFrame,
                      x_df: DataFrame,
                      y_pred: list) -> None:
    """
    This function prints all the metrics for the models.
    """

    def adjusted_r2(y_true: list, y_pred: list, n: int, p: int) -> float:
        """
        This function calculates the adjusted R2 score.
        """
        r2 = r2_score(y_true, y_pred)
        return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    mse = MeanSquaredError()
    rmse = RootMeanSquaredError()

    mse_linear = mse.calculate(y_df, y_pred[0])
    mse_bayesian = mse.calculate(y_df, y_pred[1])
    mse_decision = mse.calculate(y_df, y_pred[2])

    rmse_linear = rmse.calculate(y_df, y_pred[0])
    rmse_bayesian = rmse.calculate(y_df, y_pred[1])
    rmse_decision = rmse.calculate(y_df, y_pred[2])

    print("Mean Squared Error for Linear Regression: ")
    print(f"{mse_linear}")
    print("Mean Squared Error for Bayesian Ridge: ")
    print(f"{mse_bayesian}")
    print("Mean Squared Error for Decision Tree Regression: ")
    print(f"{mse_decision}")
    print("Mean Squared Error for Feedforward Neural Network: ")
    print(f"{mse.calculate(y_df, y_pred[3])}")
    print("")
    print("Root Mean Squared Error for Linear Regression: ")
    print(f"{rmse_linear}")
    print("Root Mean Squared Error for Bayesian Ridge: ")
    print(f"{rmse_bayesian}")
    print("Root Mean Squared Error for Decision Tree Regression: ")
    print(f"{rmse_decision}")
    print("Root Mean Squared Error for Feedforward Neural Network: ")
    print(f"{rmse.calculate(y_df, y_pred[3])}")
    print("")
    print("R2 Score for Linear Regression: ")
    print(f"{r2_score(y_df, y_pred[0])}")
    print("R2 Score for Bayesian Ridge: ")
    print(f"{r2_score(y_df, y_pred[1])}")
    print("R2 Score for Decision Tree Regression: ")
    print(f"{r2_score(y_df, y_pred[2])}")
    print("R2 Score for Feedforward Neural Network: ")
    print(f"{r2_score(y_df, y_pred[3])}")
    print("")
    print("adjusted R2 Score for Linear Regression: ")
    print(f"{adjusted_r2(y_df, y_pred[0], len(y_df), x_df.shape[1])}")
    print("adjusted R2 Score for Bayesian Ridge: ")
    print(f"{adjusted_r2(y_df, y_pred[1], len(y_df), x_df.shape[1])}")
    print("adjusted R2 Score for Decision Tree Regression: ")
    print(f"{adjusted_r2(y_df, y_pred[2], len(y_df), x_df.shape[1])}")
    print("adjusted R2 Score for Feedforward Neural Network: ")
    print(f"{adjusted_r2(y_df, y_pred[3], len(y_df), x_df.shape[1])}")
    print("")


def plot_results(models: list, y: list) -> None:
    """
    This function plots the results.
    """
    figure, axes = plt.subplots(3, 2, figsize=(10, 10))

    axes[0, 0].scatter(y, models[0], color='blue')
    axes[0, 0].set_title("Linear Regression")
    axes[0, 0].set_xlabel("True Values")
    axes[0, 0].set_ylabel("Predictions")

    axes[0, 1].scatter(y, models[1], color='red')
    axes[0, 1].set_title("Bayesian Ridge")
    axes[0, 1].set_xlabel("True Values")
    axes[0, 1].set_ylabel("Predictions")

    axes[1, 0].scatter(y, models[2], color='green')
    axes[1, 0].set_title("Decision Tree Regression")
    axes[1, 0].set_xlabel("True Values")
    axes[1, 0].set_ylabel("Predictions")

    axes[1, 1].scatter(y, models[0], color='blue',
                       label='Linear')
    axes[1, 1].scatter(y, models[1], color='red',
                       label='Bayesian')
    axes[1, 1].scatter(y, models[2], color='green',
                       label='Decision Tree')
    axes[1, 1].scatter(y, models[3], color='purple', label='FNN')
    axes[1, 1].set_title("Comparison")
    axes[1, 1].set_xlabel("True Values")
    axes[1, 1].set_ylabel("Predictions")
    axes[1, 1].legend()

    axes[2, 0].scatter(y, models[3], color='purple')
    axes[2, 0].set_title("Feedforward Neural Network")
    axes[2, 0].set_xlabel("True Values")
    axes[2, 0].set_ylabel("Predictions")

    plt.show()


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
