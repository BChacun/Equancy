import streamlit as st
import pandas as pd
import numpy as np

st.title('Algorithms :')

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
page.app()

# END Pages














#streamlit run C:\Users\bapti\Desktop\Equancy\Rendu\main.py