from datetime import time
import time
import streamlit as st
import pandas as pd
import math
from sklearn.metrics import *
from math import sqrt

# IMPORTS :

import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import *
from tensorflow.python.keras import optimizers as opt
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


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
ep_prec = None
l_prec = None


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>GRU</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Gated Recurrent Unit_</center></h1>', unsafe_allow_html=True)

    def GRU_MODEL(l, ep):

        # Split dataset

        # For analysis
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

        # transform data and fit scaler
        train_scaled = scaler.fit_transform(np.array(train_m))
        test_scaled = scaler.transform(np.array(test_m))

        # train data 
        y_train = train_scaled[:, -1]
        X_train = train_scaled[:, 0:-1]
        X_train = X_train.reshape(len(X_train), 1, 1)

        # test data
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
        with st.spinner('Training may take a while, so grab a cup of coffee, or better, go for a run!'):
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

    global results, mean_error, mae, rmse, mape, mapd, duree, change
    global data_prec
    global ep_prec, l_prec

    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    container = st.expander("View parameters")
    ep = container.number_input('Choose the number of epochs', min_value=10, max_value=100000, value=100, step=100)
    if ep_prec != ep:
        ep_prec = ep
        change = True

    l = container.number_input('Choose learning_rate', min_value=0.001, max_value=0.1, value=0.01, step=0.002)
    if l_prec != l:
        l_prec = l
        change = True

    if results is None:
        data_prec = df
        with st.spinner('Wait for it: 1st loading'):
            results, mean_error, mae, rmse, mape, mapd, duree = GRU_MODEL(l, ep)
            change = False

    elif change:
        data_prec = df
        with st.spinner('Wait for it: reloading'):
            results, mean_error, mae, rmse, mape, mapd, duree = GRU_MODEL(l, ep)
            change = False

    elif not df.equals(data_prec):
        data_prec = df
        with st.spinner('Wait for it: data has changed'):
            results, mean_error, mae, rmse, mape, mapd, duree = GRU_MODEL(l, ep)
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
