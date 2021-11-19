import streamlit as st
import pandas as pd
import numpy as np

st.title('Algorithms :')

# Création du Database

@st.cache
def load_data(n):
    data = pd.read_csv(r'C:\Users\Tddon\PJE FB Prophet\Equancy\DATABASE.txt', sep=";", header=None, na_values=['?'])

    data = data[:n]

    data = data.rename(columns=data.iloc[0]).drop(data.index[0])

    data["Time_index2"] = data["Date"] + " " + data["Time"]

    data["timestamp"] = pd.to_datetime(data['Time_index2'], format='%d/%m/%Y %H:%M:%S')
    data.set_index(['timestamp'],inplace=True)

    data["Global_active_power"] = data["Global_active_power"].astype(float)

    df = pd.DataFrame(data, columns=['Global_active_power'])

    df = df.rename(columns={'Global_active_power': 'Y'})
    
    return df

df = load_data(1000)

# Pages :
import AR
import ARIMA

PAGES = {
    "AR": AR,
    "ARIMA": ARIMA
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app(df)

# END Pages














#streamlit run C:\Users\Tddon\PJE FB Prophet\Equancy\Rendu\main.py