from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    """
    LinearRegressionModel

    This class is a wrapper for the LinearRegression model from sklearn.
    """
    def __init__(self):
        """
        initialize the model with the LinearRegression model from sklearn
        """
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """
        train the model with the training data

        Parameters
        ----------
        X_train : array-like
            The training input samples.
        y_train : array-like
            The target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        predict the target values for the input samples

        Parameters
        ----------
        X_test : array-like
            The input samples.
        """
        return self.model.predict(X_test)
