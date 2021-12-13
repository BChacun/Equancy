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


def AR(p, df):
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

    # Time of the prediction
    end = time.time()
    duree = end - start

    # Statistics
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions

    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(p, df):
    if 'results_AR' not in st.session_state:
        st.session_state.resultsAR = AR(p, df)
    if 'state_p' not in st.session_state:
        st.session_state.state_p = p
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(p, df):
    if p != st.session_state.state_p:
        st.session_state.state_p = p
        with st.spinner('Wait for it...'):
            st.session_state.resultsAR = AR(p, df)
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it...'):
            st.session_state.state_dataset = df
            st.session_state.resultsAR = AR(p, df)


def inputParameters():
    container = st.expander("View parameters")
    p = container.number_input('Choose p', min_value=1, max_value=50, value=25, step=1)
    return p


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>AR</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Auto Regressive model_</center></h1>', unsafe_allow_html=True)
    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    p = inputParameters()
    initStates(p, df)
    changeStates(p, df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsAR
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)

