from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_DNN(input_shape):
    """
    Defines and returns a deep learning model.

    Args:
        input_shape (tuple): Shape of the input data (number of features).

    Returns:
        tf.keras.Model: Compiled deep learning model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.2),  # Prevent overfitting
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
