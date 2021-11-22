import streamlit as st
import pandas as pd
import numpy as np

st.title('Algorithms :')

# Création du Database

@st.cache
def load_data(product_id, store_id):
    sales_df = pd.read_parquet('sales_data.parquet')
    data = (sales_df
 # On sélectionne le produit 6716 dans le magasin 171
 .query(f'product_id == {product_id} and store_id == {store_id}')

 # On convertit la date en datetime
 .assign(date=lambda _df: pd.to_datetime(_df['date']))

 # On calcule la date du lundi de la semaine de CAL_Date
 .assign(weekDate=lambda _df: _df['date'] - _df['date'].dt.weekday * np.timedelta64(1, 'D'))

 # On groupe les ventes à la semaine
 .groupby(['weekDate'])
 .agg(
     quantity_weekly=('quantity', 'sum')
 )
            .reset_index()
 .set_index('weekDate')
 [['quantity_weekly']]
       );
    df = pd.DataFrame(data)
    return df



df = load_data(2008,9)

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
