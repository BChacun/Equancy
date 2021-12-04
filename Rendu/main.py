import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.metrics import *
from math import sqrt
import statistics
# Pages :
import AR
import ARIMA
import SES
import HWES
import SARIMA
import Intro
import View_data
import View_benchmark


# Fonction pour créer la data (Attention, je le fais à partir des csv)
def creation_database(PRODUCT_ID, STORE_ID):
    data_init = pd.read_csv(f'Equancy\Rendu\{PRODUCT_ID}_{STORE_ID}.csv')
    data_init.head()
    data_init = data_init.assign(weekDate=lambda _df: pd.to_datetime(_df['weekDate'], format="%Y-%m-%d"))
    data = data_init.copy()
    data.set_index("weekDate", inplace=True)
    data = data.squeeze()
    return data


# Pages des modèles de prévision
PAGES = {
    "None": None,
    "SES": SES,
    "HWES": HWES,
    "AR": AR,
    "ARIMA": ARIMA,
    "SARIMA": SARIMA,
}


# Fonction pour charger la data
@st.cache
def load_data(product_id, store_id):
    data = creation_database(product_id, store_id)
    return data

# Liste des magasins et produits
Store_ids = ['09', '36']
Product_ids = ['2008', '183', '151', '101', '230']

# Bouton pour retourner à la page d'accueil
button_home = st.sidebar.button("RETURN HOME")
st.sidebar.write('--------------')
st.sidebar.title('Forecasts of sales:')

# Selectboxes pour choisir le magasin parmi Store_ids et le produit parmi Product_ids
c1, c2 = st.sidebar.columns(2)
STORE_ID = c1.selectbox("Select Store ID :", Store_ids, key='STORE_ID')
PRODUCT_ID = c2.selectbox("Select Product ID :", Product_ids, key='PRODUCT_ID')

# Bouton pour voir la data
button_view = st.sidebar.button("View Data", help='click to view data')

# Selectbox pour choisir le modèle de prévision parmi les modèles dans PAGES
selection_model = st.sidebar.selectbox("Select forecasting model :", list(PAGES.keys()),
                                       help='if None, goes back to home', key='selection')
st.sidebar.write('--------------')

# Bouton pour voir les résultats du Benchmark
button_results = st.sidebar.button("View Benchmark Results")

# st.cache(func=load_data(PRODUCT_ID, STORE_ID), allow_output_mutation=True)

# On charge la data du magasin STORE_ID pour le produit PRODUCT_ID
df = load_data(PRODUCT_ID, STORE_ID)

# modèle de prévision actuellement selectionné
page = PAGES[selection_model]

# Si on a cliqué sur le bouton "Return Home" ou si on n'a pas choisi de modèle, et qu'on a pas cliqué sur le bouton
# pour voir la data ou sur le bouton pour voir les résultats du Benchmark
if (button_home or selection_model == 'None') and not button_view and not button_results:
    # On affiche la page d'accueil
    Intro.app(df)

# Sinon, si on a cliqué sur le bouton pour voir la data
elif button_view:
    # On affiche la page View data
    View_data.app(df)

# Sinon, si on a cliqué sur le bouton pour voir les résultats du Benchmark
elif button_results:
    # On affiche la page View benchmark
    View_benchmark.app(df)

# Sinon
else:
    # Si on a choisi un modèle de prévision
    if selection_model != "None":
        # On affiche la page du modèle
        page.app(df)

# streamlit run D:\Equancy\Equancy_Local\Rendu\main.py
