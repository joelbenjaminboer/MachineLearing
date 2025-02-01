from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def build_FNN(input_shape) -> Sequential:
    """
    Defines and returns a deep learning model.

    Args:
        input_shape (tuple): Shape of the input data (number of features).

    Returns:
        keras.Model: Compiled deep learning model.
    """
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(0.01),
              input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1,  kernel_regularizer=l2(0.01))
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
