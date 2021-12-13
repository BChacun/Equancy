from datetime import time
import time
import streamlit as st
import math
from sklearn.metrics import *
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# Méthode de transformation de notre DataSet en une série supervisée pour la prévision
def series_to_supervised(df, n_in=1, n_out=1):
    cols = list()
    # Séparation
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # Concaténation
    agg = pd.concat(cols, axis=1)
    agg.dropna(axis=0, inplace=True)
    return agg.values


def train_test_split(series, frac):
    n = len(series)
    ind = math.floor(frac * n)
    X = series[:ind, :]
    Y = series[ind:, :]
    return X, Y


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


def RF(df):

    res = separate_data(df, 0.8)
    train_serie, test_serie = res[0], res[1]
    df_bis = df.reset_index()
    # Création de la série supervisée
    X = series_to_supervised(df_bis['quantity_weekly'], n_in=1, n_out=1)

    # Création des ensembles d'entrainement et de test
    X_train, X_test = train_test_split(X, 0.8)
    X_train, Y_train = X_train[:, :-1], X_train[:, -1]
    Y_test = X_test[:, :-1]

    start = time.time()
    # Entrainement à l'aide d'une Cross-Validation afin de chercher le meilleur hyper-paramètre
    param_grid = {'n_estimators': np.arange(1, 10, 1)}
    grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
    grid.fit(X_train, Y_train)

    # On récupère l'estimateur avec l'hyperparamètre le plus optimal
    model = grid.best_estimator_

    # On prédit l'avenir
    prediction = model.predict(Y_test)
    end = time.time()
    indexes = df_bis['weekDate'][-len(test_serie)-1:-1]
    duree = end - start
    test = pd.Series(data=[Y_test[i][0] for i in range(len(test_serie))], index=indexes)
    predictions = pd.Series(data=prediction, index=indexes)

    # Statistics
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions

    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(df):
    if 'results_RF' not in st.session_state:
        st.session_state.resultsRF = RF(df)
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(df):
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it...'):
            st.session_state.state_dataset = df
            st.session_state.resultsRF = RF(df)


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>RF</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Random Forest model_</center></h1>', unsafe_allow_html=True)
    initStates(df)
    changeStates(df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsRF
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
