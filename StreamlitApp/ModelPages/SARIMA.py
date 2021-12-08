from datetime import time
import time
import streamlit as st
import pandas as pd
import math
from sklearn.metrics import *
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX


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


def SARIMA(param, param_seasonal, df):
    # Split dataset
    # param = (p, d, q)
    # param_seasonal = (P, D, Q, m)
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]

    # Train model
    start = time.time()
    model = SARIMAX(train.values, order=param, seasonal_order=param_seasonal)
    model_fit = model.fit(optimized=True)

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


def initStates(param, param_seasonal, df):
    if 'results_SARIMA' not in st.session_state:
        st.session_state.resultsSARIMA = SARIMA(param, param_seasonal, df)
    if 'state_param' not in st.session_state:
        st.session_state.state_param = param
    if 'state_param_seasonal' not in st.session_state:
        st.session_state.state_param_seasonal = param_seasonal
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(param, param_seasonal, df):
    if param != st.session_state.state_param or param_seasonal != st.session_state.state_param_seasonal:
        if param != st.session_state.state_param:
            st.session_state.state_p = param
        if param_seasonal != st.session_state.state_param_seasonal:
            st.session_state.state_d = param_seasonal
        with st.spinner('Wait for it: parameters have changed...'):
            st.session_state.resultsSARIMA = SARIMA(param, param_seasonal, df)
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it: parameters have changed...'):
            st.session_state.state_dataset = df
            st.session_state.resultsSARIMA = SARIMA(param, param_seasonal, df)


def inputParameters():
    container = st.expander("View parameters")
    c1, c2 = container.columns(2)
    p = c1.number_input('Choose p', min_value=1, max_value=100, value=1, step=1)
    d = c1.number_input('Choose d', min_value=0, max_value=2, value=1, step=1)
    q = c1.number_input('Choose q', min_value=0, max_value=100, value=1, step=1)
    P = c2.number_input('Choose P', min_value=1, max_value=100, value=1, step=1)
    D = c2.number_input('Choose D', min_value=0, max_value=2, value=1, step=1)
    Q = c2.number_input('Choose Q', min_value=0, max_value=100, value=0, step=1)
    m = c2.number_input('Choose m', min_value=0, max_value=100, value=49, step=1)
    param = (p, d, q)
    param_seasonal = (P, D, Q, m)
    return param, param_seasonal


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>SARIMA</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Seasonal Autoregressive Integrated Moving Average_</center></h1>', unsafe_allow_html=True)
    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    param, param_seasonal = inputParameters()
    initStates(param, param_seasonal, df)
    changeStates(param, param_seasonal, df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsSARIMA
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
