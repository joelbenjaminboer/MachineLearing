from sklearn.linear_model import BayesianRidge


class BayesianRidgeModel:
    """
    BayesianRidgeModel

    This class is a wrapper for the BayesianRidge model from sklearn.
    """
    def __init__(self) -> None:
        """
        initialize the model with the BayesianRidge model from sklearn
        """
        self.model = BayesianRidge()

    def train(self, X_train: list, y_train: list) -> None:
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

    def predict(self, X_test: list) -> list:
        """
        predict the target values for the input samples

        Parameters
        ----------
        X_test : array-like
            The input samples.
        """
        return self.model.predict(X_test)
