from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.data_setup import merge_data, preprocess_data, split_data
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the data
df = merge_data()
data = preprocess_data(df)
train_data, test_data, valid_data = split_data(data)

# Split the data into features and target
X_train = train_data.drop(columns='price')
y_train = train_data['price']

X_valid = valid_data.drop(columns='price')
y_valid = valid_data['price']

X_test = test_data.drop(columns='price')
y_test = test_data['price']

# Split the data into training and testing sets

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Prevent overfitting
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression (single continuous value)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")
print(f"r2_score: {r2_score(y_test, model.predict(X_test))}")4
