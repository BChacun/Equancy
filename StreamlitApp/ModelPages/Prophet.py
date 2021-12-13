import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import fbprophet
import time
import streamlit as st
from math import sqrt


def separate_data(data, frac):
    ind = math.floor(frac * len(data))
    train = data[:ind]
    train = train.to_frame()
    train = train.reset_index()
    train.rename(columns={'weekDate': 'ds', 'quantity_weekly': 'y'}, inplace=True)
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


def Prophet(df):

    train, test = separate_data(df, 0.8)

    # Training
    start = time.time()
    model = fbprophet.Prophet(yearly_seasonality=True)
    model.fit(train)

    # Forecasting
    future = test.to_frame().copy()
    future = future.reset_index()
    future.rename(columns={'weekDate': 'ds', 'quantity_weekly': 'y'}, inplace=True)
    future.drop('y', axis=1, inplace=True)

    forecast = model.predict(future)
    predictions = forecast[['ds', 'yhat']]

    predictions.set_index("ds", inplace=True)
    predictions = predictions.squeeze()
    end = time.time()
    duree = end - start
    mean_error, mae, rmse, mape, mapd = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results = results.to_frame()
    results["Prediction"] = predictions

    return results, mean_error, mae, rmse, mape, mapd, duree


def initStates(df):
    if 'results_Prophet' not in st.session_state:
        st.session_state.resultsProphet = Prophet(df)
    if 'state_dataset' not in st.session_state:
        st.session_state.state_dataset = df


def changeStates(df):
    if not df.equals(st.session_state.state_dataset):
        with st.spinner('Wait for it...'):
            st.session_state.state_dataset = df
            st.session_state.resultsProphet = Prophet(df)


def app(df):
    st.markdown('<h1><center><span style="color:#00BFFF"><U>Prophet</U></center></h1>', unsafe_allow_html=True)
    st.markdown('<h0><center>_Facebook Prophet_</center></h1>', unsafe_allow_html=True)
    initStates(df)
    changeStates(df)
    results, mean_error, mae, rmse, mape, mapd, duree = st.session_state.resultsProphet
    plot_streamlit(results, mean_error, mae, rmse, mape, mapd, duree)
