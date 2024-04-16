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
    ax.plot(df)  # Assuming the data has columns for x and y values

    # (Optional) Customize your plot using ax methods
    # For example, set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series Plot")

    # Display the plot
    st.pylot(fig)    

if __name__ == '__main__':
    app()   


