from models.Baysian_regression import BayesianRidgeModel
from models.Linear_regression import LinearRegressionModel
from models.Decision_tree_regression import DecisionTreeRegressionModel
from data.data_setup import merge_data
from data.data_setup import split_data
from data.data_setup import preprocess_data
from metrics.Mean_squared_error import MeanSquaredError
from metrics.Root_mean_squared_error import RootMeanSquaredError


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

    print(f"Mean Squared Error for Linear Regression:"
          f"{mse_linear}")
    print(f"Mean Squared Error for Bayesian Ridge:"
          f"{mse_bayesian}")
    print(f"Mean Squared Error for Decision Tree Regression:"
          f"{mse_decision}")

    print(f"Root Mean Squared Error for Linear Regression:"
          f"{rmse_linear}")
    print(f"Root Mean Squared Error for Bayesian Ridge:"
          f"{rmse_bayesian}")
    print("Root Mean Squared Error for Decision Tree Regression:"
          f"{rmse_decision}")


if __name__ == "__main__":
    train()
