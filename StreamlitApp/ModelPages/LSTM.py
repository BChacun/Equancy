from datetime import time
import time
import streamlit as st
import pandas as pd
import math
from sklearn.metrics import *
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


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
    st.write("Le modèle LSTM de Pytorch donne de mauvais résultats car les hyperparamètres n'ont pas été optimisés et "
             "le modèle a été peu entraîné")
    st.markdown('<h5><U>Indicators :</U></h5>', unsafe_allow_html=True)
    st.markdown('<li><em>Mean Error : </em>%.3f</li>' % mean_error, unsafe_allow_html=True)
    st.write('<li><em>MAE : </em>%.3f</li>' % mae, unsafe_allow_html=True)
    st.write('<li><em>RMSE : </em>%.3f</li>' % rmse, unsafe_allow_html=True)
    st.write(f'<li><em>MAPE : </em>{round(mape, 3)} </li>', unsafe_allow_html=True)
    st.write(f'<li><em>MAPD : </em>{round(mapd, 3)} </li>', unsafe_allow_html=True)
    st.write('<li><em>Temps de calcul de la prévision : </em>%.3f s</li>' % duree, unsafe_allow_html=True)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def LSTM_MODEL(l, ep, df):

    # For analysis
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]
    len_test = len(test)
    train_window = len_test

    test_data = test.values.astype(float)
    train_data = train.values.astype(float)

    # Data Normalization :
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    # Conversion for Pytorch :
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=l)

    start = time.time()
    ep = int(ep)
    for i in range(ep):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    test_inputs = test.tolist()

    # Prédiction lent(test) -> 1

    model.eval()

    for i in range(len_test):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
            actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

    # Make predictions

    predictions = pd.Series(actual_predictions.ravel(), index=test.index)
    end = time.time()
    duree = end - start
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions

    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(l, ep, df):
    if 'results_LSTM' not in st.session_state:
        st.session_state.resultsLSTM = LSTM_MODEL(l, ep, df)
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
            st.session_state.resultsGRU = LSTM_MODEL(l, ep, df)
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it...'):
            st.session_state.state_dataset = df
            st.session_state.resultsARIMA = LSTM_MODEL(l, ep, df)


def inputParameters():
    container = st.expander("View parameters")
    ep = container.number_input('Choose the number of epochs', min_value=1, max_value=100000, value=10, step=100)
    l = container.number_input('Choose learning_rate', min_value=0.0001, max_value=0.1, value=0.01, step=0.002)
    return ep, l


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>LSTM Pytorch</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Long Short-Term Memory_</center></h1>', unsafe_allow_html=True)
    st.markdown('<h5><U>Parameters :</U></h5>', unsafe_allow_html=True)
    ep, l = inputParameters()
    initStates(l, ep, df)
    changeStates(l, ep, df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsLSTM
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
