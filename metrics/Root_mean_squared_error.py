from sklearn.metrics import mean_squared_error


class RootMeanSquaredError:
    """
    RootMeanSquaredError

    This class is a wrapper for the mean_squared_error function from sklearn.
    """
    def __init__(self) -> None:
        """
        initialize the metric with the mean_squared_error function from sklearn
        """
        self.metric = mean_squared_error

    def calculate(self, y_true: list, y_pred: list) -> float:
        """
        calculate the mean squared error for the true and predicted values

        Parameters
        ----------
        y_true : array-like
            The true target values.
        y_pred : array-like
            The predicted target values.
        """
        return self.metric(y_true, y_pred)
