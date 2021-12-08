import streamlit as st


def app(df, Product_ID, Store_ID):
    st.markdown(f'<h3><center>Ventes par semaine du produit {Product_ID} dans le magasin {Store_ID} :</center></h3>', unsafe_allow_html=True)
    st.write('---------------------------------------------------------------------------------------')
    st.line_chart(data=df, width=0, height=0, use_container_width=True)
    st.write('---------------------------------------------------------------------------------------')
