import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def app():
    st.title('Time Series Analysis')

    # Load the data
    df = pd.read_csv('https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv', header=0)
    st.write(df)
    st.write(df.shape)

    # Create a figure and axes using plt.subplots
    fig, ax = plt.subplots()

    # Use ax.plot to plot the data from your DataFrame
    ax.plot(df['#Passengers'])  # Assuming the data has columns for x and y values

    # (Optional) Customize your plot using ax methods
    # For example, set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series Plot")

    # Display the plot
    st.pyplot(fig)    

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(df.iloc[:,1].values.reshape(-1, 1))

    # Split the data into input and output sequences
    window_size = 12
    input_data = []
    output_data = []
    for i in range(len(data_norm)-window_size):
        input_data.append(data_norm[i:i+window_size])
        output_data.append(data_norm[i+window_size])

    # Convert the data to numpy arrays
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    # Split the data into training and testing sets
    split_index = int(len(input_data) * 0.8)
    x_train = input_data[:split_index]
    y_train = output_data[:split_index]
    x_test = input_data[split_index:]

    y_test = output_data[split_index:]

    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=128, input_shape=(window_size, 1)),
    tf.keras.layers.Dense(units=1)
])
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_test, y_test))

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot training and validation loss on the same axes
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')

    # Customize the plot using ax methods
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()  # Legend placement can be adjusted with optional arguments

    # Display the plot
    st.pyplot




    

if __name__ == '__main__':
    app()   


