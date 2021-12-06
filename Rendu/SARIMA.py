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


results, mean_error, mae, rmse, mape, mapd, duree = None, None, None, None, None, None, None
data_prec = None
p_prec, d_prec, q_prec,  P_prec, D_prec, Q_prec, m_prec = None, None, None, None, None, None, None
change = False


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>SARIMA</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Seasonal Autoregressive Integrated Moving Average_</center></h1>', unsafe_allow_html=True)

    def SARIMA(param, param_seasonal):
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

    global results, mean_error, mae, rmse, mape, mapd, duree
    global data_prec
    global p_prec, d_prec, q_prec,  P_prec, D_prec, Q_prec, m_prec
    global change

    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    container = st.expander("View parameters")
    c1, c2 = container.columns(2)
    p = c1.number_input('Choose p', min_value=1, max_value=100, value=1, step=1)
    if p_prec != p:
        p_prec = p
        change = True
    d = c1.number_input('Choose d', min_value=0, max_value=2, value=1, step=1)
    if d_prec != d:
        d_prec = d
        change = True
    q = c1.number_input('Choose q', min_value=0, max_value=100, value=1, step=1)
    if q_prec != q:
        q_prec = q
        change = True

    P = c2.number_input('Choose P', min_value=1, max_value=100, value=1, step=1)
    if P_prec != P:
        P_prec = P
        change = True

    D = c2.number_input('Choose D', min_value=0, max_value=2, value=1, step=1)
    if D_prec != D:
        D_prec = D
        change = True

    Q = c2.number_input('Choose Q', min_value=0, max_value=100, value=0, step=1)
    if Q_prec != Q:
        Q_prec = Q
        change = True

    m = c2.number_input('Choose m', min_value=0, max_value=100, value=49, step=1)
    if m_prec != m:
        m_prec = m
        change = True

    param = (p, d, q)
    param_seasonal = (P, D, Q, m)

    if results is None:
        data_prec = df
        with st.spinner('Wait for it: 1st loading'):
            results, mean_error, mae, rmse, mape, mapd, duree = SARIMA(param, param_seasonal)
            change = False

    elif change:
        data_prec = df
        with st.spinner('Wait for it: reloading'):
            results, mean_error, mae, rmse, mape, mapd, duree = SARIMA(param, param_seasonal)
            change = False

    elif not df.equals(data_prec):
        data_prec = df
        with st.spinner('Wait for it: data has changed'):
            results, mean_error, mae, rmse, mape, mapd, duree = SARIMA(param, param_seasonal)
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
