import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('C:\Users\yesha\OneDrive\Stockpred\all_stocks_5yr.csv.zip')
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()
st.title("📈 Stock Price Prediction App")
st.write("Predict future stock close prices using TensorFlow LSTM")
companies = data['Name'].unique()
company = st.selectbox("Select a company:", companies)

company_data = data[data['Name'] == company]
st.line_chart(company_data[['date','close']].set_index('date'))
close_data = company_data.filter(['close']).values
training_size = int(np.ceil(len(close_data) * 0.95))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

train_data = scaled_data[0:training_size,:]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(32),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10)
test_data = scaled_data[training_size-60:,:]
x_test, y_test = [], close_data[training_size:,:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

st.subheader("Predicted vs Actual Close Prices")
fig, ax = plt.subplots()
ax.plot(company_data['date'][training_size:], y_test, label="Actual")
ax.plot(company_data['date'][training_size:], predictions, label="Predicted")
ax.legend()
st.pyplot(fig)
future_input = scaled_data[-60:].tolist()
future_predictions = []

for _ in range(5*365):  # 5 years daily prediction
    x_future = np.array([future_input[-60:]])
    x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
    pred = model.predict(x_future)
    future_input.append(pred[0])
    future_predictions.append(scaler.inverse_transform(pred)[0][0])

st.subheader("5-Year Future Prediction")
st.line_chart(future_predictions) 