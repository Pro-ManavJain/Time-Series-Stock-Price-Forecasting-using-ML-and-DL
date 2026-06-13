import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#downloading stock data
df = yf.download(
    tickers="AAPL",
    start="2018-01-01",
    end="2024-01-01",
    interval="1d"
)
print("first 5 rows:")
print(df.head())

print("\ndata shape:")
print(df.shape)

print("\ncolumns:")
print(df.columns)

#converting multiIndex columns to single index
df.columns = df.columns.get_level_values(0)
print("\nupdated columns:")
print(df.columns)
#sorting data
df = df.sort_index()
target_column = "Close"

print(df.head())
print(df.tail())

#input
X = df[['Open', 'High', 'Low', 'Volume']]
# output
y = df['Close']

#training data
train_size = int(len(df) * 0.8)
X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test  = y.iloc[train_size:]

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1))

print("Scaled X range:", X_train_scaled.min(), X_train_scaled.max())
print("Scaled y range:", y_train_scaled.min(), y_train_scaled.max())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#linear model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled)

#predicting the scaled data
y_pred_lr_scaled = lr_model.predict(X_test_scaled)

y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_lr))
mae = mean_absolute_error(y_test_actual, y_pred_lr)

print("Linear Regression RMSE:", rmse)
print("Linear Regression MAE :", mae)

#non-linear model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train_scaled.ravel())

y_pred_rf_scaled = rf_model.predict(X_test_scaled)
y_pred_rf = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1, 1))


rmse_rf = np.sqrt(mean_squared_error(y_test_actual, y_pred_rf))
mae_rf = mean_absolute_error(y_test_actual, y_pred_rf)

print("Random Forest RMSE:", rmse_rf)
print("Random Forest MAE :", mae_rf)

#LSTM training
TIME_STEPS = 60


def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []

    for i in range(time_steps, len(X)):    #takes previous 60 days
        X_seq.append(X[i - time_steps:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

X_train_lstm, y_train_lstm = create_sequences(      #training
    X_train_scaled,
    y_train_scaled,
    TIME_STEPS
)
X_test_lstm, y_test_lstm = create_sequences(        #testing
    X_test_scaled,
    y_test_scaled,
    TIME_STEPS
)

print("X_train_lstm shape:", X_train_lstm.shape)
print("y_train_lstm shape:", y_train_lstm.shape)
print("X_test_lstm shape :", X_test_lstm.shape)
print("y_test_lstm shape :", y_test_lstm.shape)

#DL training
#importing libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

lstm_model = Sequential()

lstm_model.add(
    LSTM(
        units=50,
        activation='tanh',
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    )
)

lstm_model.add(Dense(1))

lstm_model.compile(     #compiling
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

history_lstm = lstm_model.fit(      #training
    X_train_lstm,
    y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predict on test data (scaled)
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)

# Inverse scaling to get real prices
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_test_lstm_actual = scaler_y.inverse_transform(y_test_lstm)

# LSTM metrics
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_actual, y_pred_lstm))
mae_lstm = mean_absolute_error(y_test_lstm_actual, y_pred_lstm)

print("\nLSTM RMSE:", rmse_lstm)
print("LSTM MAE :", mae_lstm)

#lstm visualising
plt.figure(figsize=(12, 5))
plt.plot(y_test_lstm_actual, label="Actual Price")
plt.plot(y_pred_lstm, label="LSTM Predicted Price")
plt.title("LSTM: Actual vs Predicted Closing Price")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()

#GRU
from tensorflow.keras.layers import GRU

gru_model = Sequential()

gru_model.add(
    GRU(
        units=50,
        activation='tanh',
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    )
)

gru_model.add(Dense(1))

gru_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

history_gru = gru_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

y_pred_gru_scaled = gru_model.predict(X_test_lstm)
y_pred_gru = scaler_y.inverse_transform(y_pred_gru_scaled)

rmse_gru = np.sqrt(mean_squared_error(y_test_lstm_actual, y_pred_gru))
mae_gru = mean_absolute_error(y_test_lstm_actual, y_pred_gru)

print("\nGRU RMSE:", rmse_gru)
print("GRU MAE :", mae_gru)
#gru visualising
plt.figure(figsize=(12, 5))
plt.plot(y_test_lstm_actual, label="Actual Price")
plt.plot(y_pred_gru, label="GRU Predicted Price")
plt.title("GRU: Actual vs Predicted Closing Price")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()

#comparison between all
print("\n== FINAL MODEL COMPARISON ==\n")
print(f"Linear Regression  -> RMSE: {rmse:.4f} | MAE: {mae:.4f}")
print(f"Random Forest      -> RMSE: {rmse_rf:.4f} | MAE: {mae_rf:.4f}")
print(f"LSTM               -> RMSE: {rmse_lstm:.4f} | MAE: {mae_lstm:.4f}")
print(f"GRU                -> RMSE: {rmse_gru:.4f} | MAE: {mae_gru:.4f}")

print("\nConclusion:")
print("• Linear Regression works as a strong baseline.")
print("• Random Forest struggles due to lack of temporal modeling.")
print("• LSTM and GRU capture time dependencies effectively.")
print("• GRU is computationally lighter, LSTM slightly more expressive.")
