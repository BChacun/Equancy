from datetime import time
import time
import streamlit as st
import pandas as pd
import math
from sklearn.metrics import *
from math import sqrt
import numpy as np
from tensorflow import keras

from numpy import asarray
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot


def separate_data(data, frac):
    ind = math.floor(frac * len(data))
    train = data[:ind]
    test = data[ind:]
    return train, test


def indicateurs_stats(test, predictions):
    mean_error = predictions.mean() - test.mean()
    mae = mean_absolute_error(test, predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    mapd = sum(abs(predictions[t] - test[t]) for t in range(len(test))) / sum(abs(test[t]) for t in range(len(test)))
    return mean_error, mae, rmse, mape, mapd


def plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree):
    st.markdown('<h5><U>Results :</U></h5>', unsafe_allow_html=True)
    st.write(results)
    # Plots :
    st.line_chart(data=results, width=0, height=0, use_container_width=True)
    st.markdown('<h5><U>Indicators :</U></h5>', unsafe_allow_html=True)
    st.markdown('<li><em>Mean Error : </em>%.3f</li>' % mean_error, unsafe_allow_html=True)
    st.write('<li><em>MAE : </em>%.3f</li>' % mae, unsafe_allow_html=True)
    st.write('<li><em>RMSE : </em>%.3f</li>' % rmse, unsafe_allow_html=True)
    st.write(f'<li><em>MAPE : </em>{round(mape, 3)} </li>', unsafe_allow_html=True)
    st.write(f'<li><em>MAPD : </em>{round(mapd, 3)} </li>', unsafe_allow_html=True)
    st.write('<li><em>Temps de calcul de la prévision : </em>%.3f s</li>' % duree, unsafe_allow_html=True)


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def XGB_MODEL(n, df):
    n = int(n)

    # Split dataset
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]
    len_test = len(test)

    # transform the time series data into supervised learning

    train_for_supervised = series_to_supervised(train.values, n_in=6)

    test_values = train.values

    # test_values = series_to_supervised(test_values.values, n_in=6)

    # split into input and output columns
    trainX, trainy = train_for_supervised[:, :-1], train_for_supervised[:, -1]

    start = time.time()
    # fit model

    model = XGBRegressor(objective='reg:squarederror', n_estimators=n)
    model.fit(trainX, trainy)

    for i in range(len_test):
        # construct an input for a new preduction
        indice = len_test - i
        row = test_values[-6:]

        # make a one-step prediction
        np.append(test_values, model.predict(asarray([row]))[0])

    # Make predictions
    pred = test_values[-len_test:]
    predictions = pd.Series(pred, index=test.index)
    end = time.time()
    duree = end - start
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions

    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(n, df):
    if 'results_XGB' not in st.session_state:
        st.session_state.resultsXGB = XGB_MODEL(n, df)
    if 'state_n' not in st.session_state:
        st.session_state.state_n = n
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(n, df):
    if n != st.session_state.state_n:
        if n != st.session_state.state_n:
            st.session_state.state_n = n
        with st.spinner('Wait for it: parameters have changed...'):
            st.session_state.resultsXGB = XGB_MODEL(n, df)
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it: parameters have changed...'):
            st.session_state.state_dataset = df
            st.session_state.resultsARIMA = XGB_MODEL(n, df)


def inputParameters():
    container = st.expander("View parameters")

    n = container.number_input('Choose n_estimators', min_value=10, max_value=100000, value=1000, step=100)
    return n


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>XGB</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_XGBoost_</center></h1>', unsafe_allow_html=True)
    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    n = inputParameters()
    initStates(n, df)
    changeStates(n, df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsXGB
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
