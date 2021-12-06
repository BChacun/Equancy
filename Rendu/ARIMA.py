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


results, mean_error, mae, rmse, mape, mapd, duree = None, None, None, None, None, None, None
data_prec = None
p_prec, d_prec, q_prec = None, None, None
change = False


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>ARIMA</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Autoregressive Integrated Moving Average model_</center></h1>', unsafe_allow_html=True)

    def ARIMA(p, d, q):

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

    global results, mean_error, mae, rmse, mape, mapd, duree
    global data_prec
    global p_prec, d_prec, q_prec, change

    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    container = st.expander("View parameters")
    p = container.number_input('Choose p', min_value=1, max_value=100, value=30, step=1)
    if p_prec != p:
        p_prec = p
        change = True
    d = container.number_input('Choose d', min_value=0, max_value=2, value=1, step=1)
    if d_prec != d:
        d_prec = d
        change = True
    q = container.number_input('Choose q', min_value=0, max_value=100, value=10, step=1)
    if q_prec != q:
        q_prec = q
        change = True

    if results is None:
        data_prec = df
        with st.spinner('Wait for it: 1st loading'):
            results, mean_error, mae, rmse, mape, mapd, duree = ARIMA(p, d, q)
            change = False

    elif change:
        data_prec = df
        with st.spinner('Wait for it: reloading'):
            results, mean_error, mae, rmse, mape, mapd, duree = ARIMA(p, d, q)
            change = False

    elif not df.equals(data_prec):
        data_prec = df
        with st.spinner('Wait for it: data has changed'):
            results, mean_error, mae, rmse, mape, mapd, duree = ARIMA(p, d, q)
            change = False

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

    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
