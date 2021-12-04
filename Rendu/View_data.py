import streamlit as st


def app(df):
    st.line_chart(data=df, width=0, height=0, use_container_width=True)
