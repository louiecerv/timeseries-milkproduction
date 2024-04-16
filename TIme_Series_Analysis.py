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

if __name__ == '__main__':
    app()   


