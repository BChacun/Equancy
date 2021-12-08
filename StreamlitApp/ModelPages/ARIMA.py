from datetime import time
import time
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
import pandas as pd
import math
from sklearn.metrics import *
from math import sqrt


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


def ARIMA(p, d, q, df):

    # Split dataset
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]

    # Train model
    start = time.time()
    model = ARIMA_model(train.values, order=(p, d, q))
    model_fit = model.fit()

    # Make predictions
    prediction = model_fit.forecast(len(test))
    predictions = pd.Series(prediction, index=test.index)
    end = time.time()
    duree = end - start
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)
    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions
    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(p, d, q, df):
    if 'results_ARIMA' not in st.session_state:
        st.session_state.resultsARIMA = ARIMA(p, d, q, df)
    if 'state_p' not in st.session_state:
        st.session_state.state_p = p
    if 'state_d' not in st.session_state:
        st.session_state.state_d = d
    if 'state_q' not in st.session_state:
        st.session_state.state_q = q
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(p, d, q, df):
    if p != st.session_state.state_p or d != st.session_state.state_d or q != st.session_state.state_q:
        if p != st.session_state.state_p:
            st.session_state.state_p = p
        if d != st.session_state.state_d:
            st.session_state.state_d = d
        if q != st.session_state.state_q:
            st.session_state.state_q = q
        with st.spinner('Wait for it: parameters have changed...'):
            st.session_state.resultsARIMA = ARIMA(p, d, q, df)
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it: parameters have changed...'):
            st.session_state.state_dataset = df
            st.session_state.resultsARIMA = ARIMA(p, d, q, df)


def inputParameters():
    container = st.expander("View parameters")
    p = container.number_input('Choose p', min_value=1, max_value=100, value=30, step=1)
    d = container.number_input('Choose d', min_value=0, max_value=2, value=1, step=1)
    q = container.number_input('Choose q', min_value=0, max_value=100, value=10, step=1)
    return p, d, q


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>ARIMA</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Autoregressive Integrated Moving Average model_</center></h1>', unsafe_allow_html=True)
    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    p, d, q = inputParameters()
    initStates(p, d, q, df)
    changeStates(p, d, q, df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsARIMA
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
