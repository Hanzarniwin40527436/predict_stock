# Imports
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import data
data = pd.read_csv('data/data.csv')

# Drop date variable
data = data.drop(['DATE'], axis=1)

# Make data a np.array
data = data.values

# Split into train and test data (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Scale data using StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Build X and y for training and testing
X_train = train_data[:, 1:]
y_train = train_data[:, 0]
X_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Reshape input data to be 3D for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Number of features in the data
n_features = X_train.shape[2]

# Define LSTM model
def build_lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(1024, return_sequences=True, input_shape=(1, n_features)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))  # Output layer for stock price prediction
    model.compile(optimizer='adam', loss='mse')
    return model

# Create the model
model = build_lstm_model()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Save the best model
checkpoint = ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=256, 
                    callbacks=[early_stopping, checkpoint], verbose=1)

# Evaluate on test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss}")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot actual vs. predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices', linestyle='dashed')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Scaled Stock Price')
plt.legend()
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.show()
