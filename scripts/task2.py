import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv('../src/gmf_assets.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
data = df['Close']

# Split train/test by date (train: 2015–2023, test: 2024–2025)
train = data[:'2023']
test = data['2024':]

# -------------------------------
# ARIMA MODEL
# -------------------------------

print("Fitting ARIMA model...")
auto_model = pm.auto_arima(train, seasonal=False, trace=True)
order = auto_model.order
print(f"Best ARIMA order: {order}")

model_arima = ARIMA(train, order=order)
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=len(test))
forecast_arima.index = test.index

# -------------------------------
# LSTM MODEL
# -------------------------------

print("Preparing data for LSTM...")
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))

# Sequence preparation
def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

seq_length = 60
X_train_lstm, y_train_lstm = create_sequences(scaled_train, seq_length)

# Reshape for LSTM
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

print("Training LSTM model...")
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=10)
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

# Prepare test data for prediction
total_data = pd.concat([train, test])
scaled_total = scaler.transform(total_data.values.reshape(-1, 1))

X_test_lstm = []
for i in range(len(train), len(total_data)):
    past_seq = scaled_total[i - seq_length:i]
    X_test_lstm.append(past_seq)

X_test_lstm = np.array(X_test_lstm).reshape(len(X_test_lstm), seq_length, 1)

forecast_lstm_scaled = model_lstm.predict(X_test_lstm)
forecast_lstm = scaler.inverse_transform(forecast_lstm_scaled)
forecast_lstm = pd.Series(forecast_lstm.flatten(), index=test.index)

# -------------------------------
# Evaluation
# -------------------------------

def evaluate(true, pred, label='Model'):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"{label} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%\n")
    return mae, rmse, mape

mae_arima, rmse_arima, mape_arima = evaluate(test, forecast_arima, "ARIMA")
mae_lstm, rmse_lstm, mape_lstm = evaluate(test, forecast_lstm, "LSTM")

# -------------------------------
# Plotting
# -------------------------------

plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Actual')
plt.plot(forecast_arima.index, forecast_arima, label='ARIMA Forecast')
plt.plot(forecast_lstm.index, forecast_lstm, label='LSTM Forecast')
plt.title('Tesla Stock Price Forecast (2024-2025)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
