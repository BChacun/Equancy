from datetime import time
import time
import streamlit as st
import pandas as pd
import math
from sklearn.metrics import *
from math import sqrt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense


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


def GRU_MODEL(l, ep, df):
    ep = int(ep)
    # Split dataset
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]
    len_test = len(test)

    # For the model
    shifted_df = df.shift()
    concat_df = [df, shifted_df]
    data = pd.concat(concat_df, axis=1)
    data.fillna(0, inplace=True)
    res = separate_data(data, 0.8)
    train_m, test_m = res[0], res[1]

    # Scaler used
    scaler = MinMaxScaler()

    # Transform data and fit scaler
    train_scaled = scaler.fit_transform(np.array(train_m))
    test_scaled = scaler.transform(np.array(test_m))
    y_train = train_scaled[:, -1]
    X_train = train_scaled[:, 0:-1]
    X_train = X_train.reshape(len(X_train), 1, 1)
    y_test = test_scaled[:, -1]
    X_test = test_scaled[:, 0:-1]

    # Model :
    model = Sequential()

    model.add(GRU(75, input_shape=(1, 1)))
    model.add(Dense(2))
    opt = keras.optimizers.Adam(learning_rate=l)  # learning_rate=0.01
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=100, batch_size=20, shuffle=False)

    start = time.time()
    model.fit(X_train, y_train, epochs=ep, batch_size=20, shuffle=False)
    X_test = X_test.reshape(len_test, 1, 1)
    y_pred = model.predict(X_test)
    predictions_trad = scaler.inverse_transform(y_pred)
    predictions_col = []
    for i in predictions_trad:
        predictions_col.append(i[0])

    # Make predictions
    predictions = pd.Series(predictions_col, index=test.index)
    end = time.time()
    duree = end - start
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions

    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(l, ep, df):
    if 'results_GRU' not in st.session_state:
        st.session_state.resultsGRU = GRU_MODEL(l, ep, df)
    if 'state_l' not in st.session_state:
        st.session_state.state_l = l
    if 'state_ep' not in st.session_state:
        st.session_state.state_ep = ep
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(l, ep, df):
    if l != st.session_state.state_l or ep != st.session_state.state_ep:
        if l != st.session_state.state_l:
            st.session_state.state_l = l
        if ep != st.session_state.state_ep:
            st.session_state.state_ep = ep
        with st.spinner('Wait for it...'):
            st.session_state.resultsGRU = GRU_MODEL(l, ep, df)
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it...'):
            st.session_state.state_dataset = df
            st.session_state.resultsARIMA = GRU_MODEL(l, ep, df)


def inputParameters():
    container = st.expander("View parameters")
    ep = container.number_input('Choose the number of epochs', min_value=10, max_value=100000, value=100, step=100)
    l = container.number_input('Choose learning_rate', min_value=0.001, max_value=0.100, value=0.010, step=0.002)
    return ep, l


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>GRU</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Gated Recurrent Unit_</center></h1>', unsafe_allow_html=True)
    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    ep, l = inputParameters()
    initStates(l, ep, df)
    changeStates(l, ep, df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsGRU
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
