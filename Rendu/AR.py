from datetime import time
import time
import streamlit as st
from statsmodels.tsa.ar_model import AutoReg

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


results, mean_error, mae, rmse, mape, mapd, duree, p_prec = None, None, None, None, None, None, None, None
change = False
data_prec = None


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>AR</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Auto Regressive model_</center></h1>', unsafe_allow_html=True)

    def AR(p):
        # Split dataset
        res = separate_data(df, 0.8)
        train, test = res[0], res[1]

        # Train model
        start = time.time()
        model = AutoReg(train.values, p)
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

    global results, mean_error, mae, rmse, mape, mapd, duree, change, p_prec
    global data_prec

    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    p = st.number_input('Choose p', min_value=1, max_value=50, value=25, step=1)
    if p_prec != p:
        p_prec = p
        change = True

    if results is None:
        data_prec = df
        with st.spinner('Wait for it: 1st loading'):
            results, mean_error, mae, rmse, mape, mapd, duree = AR(p)
            change = False

    elif change:
        data_prec = df
        with st.spinner('Wait for it: reloading'):
            results, mean_error, mae, rmse, mape, mapd, duree = AR(p)
            change = False

    elif not df.equals(data_prec):
        data_prec = df
        with st.spinner('Wait for it: data has changed'):
            results, mean_error, mae, rmse, mape, mapd, duree = AR(p)
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
