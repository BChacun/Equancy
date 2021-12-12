import streamlit as st
import pandas as pd
from ModelPages import AR, ARIMA, GRU, HWES, LSTM, Prophet, SES, SARIMA, XGBoost
from OtherPages import Intro, View_benchmark, View_data


# Fonction pour charger la data à partir d'un CSV (les CSV ont été pré-générés à partir du fichier
# sales_data.parquet, pour éviter que le chargement au démarrage prenne trop de temps)
def load_data(PRODUCT_ID, STORE_ID):
    data_init = pd.read_csv(f'DataCSV\{PRODUCT_ID}_{STORE_ID}.csv')
    data_init.head()
    data_init = data_init.assign(weekDate=lambda _df: pd.to_datetime(_df['weekDate'], format="%Y-%m-%d"))
    data = data_init.copy()
    data.set_index("weekDate", inplace=True)
    data = data.squeeze()
    return data


@st.cache
# Fonction pour enregistrer les DataFrame des différentes combinaisons dans un dictionnaire
# Grâce au @st.cache, le résultat de cette fonction va être calculé 1 seule fois au début, puis stocké
def storageData(Product_ids, Store_ids):
    dataDict = {}
    for PRODUCT_ID in Product_ids:
        dataDict[PRODUCT_ID] = {}
        for STORE_ID in Store_ids:
            df = load_data(PRODUCT_ID, STORE_ID)
            dataDict[PRODUCT_ID][STORE_ID] = df
    return dataDict


# Pages des différents modèles de prévision
@st.cache
def buildPages():
    pages = {
        "None": None,
        "SES": SES,
        "HWES": HWES,
        "AR": AR,
        "ARIMA": ARIMA,
        "SARIMA": SARIMA,
        "GRU": GRU,
        "LSTM": LSTM,
        "Prophet": Prophet,
        "XGBoost": XGBoost,
    }
    return pages


# Fonction qui retourne le choix de l'utilisateur sur le Magasin et le Produit
def choiceStoreProduct(Product_ids, Store_ids):
    # Selectboxes pour choisir le magasin parmi Store_ids et le produit parmi Product_ids
    c1, c2 = st.sidebar.columns(2)
    STORE_ID = c1.selectbox("Store ID :", Store_ids)
    PRODUCT_ID = c2.selectbox("Product ID :", Product_ids)
    return STORE_ID, PRODUCT_ID


# Renvoie True si l'utilisateur a appuyé sur le bouton "RETURN HOME"
def homeButton():
    # Bouton pour retourner à la page d'accueil
    button_home = st.sidebar.button("RETURN HOME")
    return button_home


# Renvoie les valeurs des boutons "View Data" et "View Forecasts"
def viewButtons():
    c1, c2 = st.sidebar.columns(2)
    # Bouton pour voir la data
    buttonDataView = c1.button("View Data", help='click to view data')
    buttonForecastView = c2.button("View Forecasts", help='click to view forecasts')
    return buttonDataView, buttonForecastView


# Renvoie la String correspondant au nom du modèle choisi par l'utilisateur
def choiceModel(pages):
    # Selectbox pour choisir le modèle de prévision parmi les modèles dans PAGES
    model_choice = st.sidebar.selectbox("Forecasting model :", list(pages.keys()),
                                        help='if None, goes back to home')
    return model_choice


# Renvoie True si l'utilisateur a appuyé sur le bouton "Benchmark Results"
def benchButton():
    # Bouton pour voir les résultats du Benchmark
    button_results = st.sidebar.button("Benchmark Results")
    return button_results


# Construit la sidebar et renvoie les valeurs de ses différents éléments
def buildSidebar(pages, Product_ids, Store_ids):
    button_home = homeButton()
    st.sidebar.markdown('<h2><span style="color:#00BFFF">Forecasts of Sales :</h2>', unsafe_allow_html=True)
    STORE_ID, PRODUCT_ID = choiceStoreProduct(Product_ids, Store_ids)
    model_choice = choiceModel(pages)
    st.sidebar.write('--------------')
    buttonDataView, buttonForecastView = viewButtons()
    st.sidebar.write('--------------')

    button_results = benchButton()
    return button_home, STORE_ID, PRODUCT_ID, buttonDataView, buttonForecastView, model_choice, button_results


# Définit une variable pour chaque widget et lui affecte une valeur initiale
# L'utilisation de st.session_state permet de sauvegarder les valeurs de ces variables lors de runs successifs
# (A chaque fois qu'un widget est actionné, toutes les valeurs des widgets sont réinitialisées)
def initStates():
    if 'valH' not in st.session_state:
        st.session_state.valH = True
    if 'valS' not in st.session_state:
        st.session_state.valS = Store_ids[0]
    if 'valP' not in st.session_state:
        st.session_state.valP = Product_ids[0]
    if 'valDV' not in st.session_state:
        st.session_state.valDV = False
    if 'valFV' not in st.session_state:
        st.session_state.valFV = False
    if 'valmc' not in st.session_state:
        st.session_state.valmc = 'None'
    if 'valR' not in st.session_state:
        st.session_state.valR = False
    if 'valDF' not in st.session_state:
        st.session_state.valDF = dataDict[Product_ids[0]][Store_ids[0]]
    if 'valM' not in st.session_state:
        st.session_state.valM = None


# Change les valeurs des variables associées aux widget de STORE_ID, PRODUCT_ID et model_choice,
# lorsque l'utilisateur change les valeurs des widgets associés
def changeStates(STORE_ID, PRODUCT_ID, model_choice, pages, dataDict):
    if st.session_state.valS != STORE_ID:
        st.session_state.valS = STORE_ID
        st.session_state.valDF = dataDict[PRODUCT_ID][STORE_ID]
    if st.session_state.valP != PRODUCT_ID:
        st.session_state.valP = PRODUCT_ID
        st.session_state.valDF = dataDict[PRODUCT_ID][STORE_ID]
    if st.session_state.valmc != model_choice:
        st.session_state.valmc = model_choice
        st.session_state.valM = pages[model_choice]


# Met à jour les valeurs des variables associées aux widget lors d'un click sur le bouton "RETURN HOME"
def onClickHome():
    st.session_state.valH = True
    st.session_state.valDV = False
    st.session_state.valFV = False
    st.session_state.valmc = 'None'
    st.session_state.valR = False


# Met à jour les valeurs des variables associées aux widget lors d'un click sur le bouton "View Data"
def onClickData():
    st.session_state.valH = False
    st.session_state.valDV = True
    st.session_state.valFV = False
    st.session_state.valR = False


# Met à jour les valeurs des variables associées aux widget lors d'un click sur le bouton "View Forecasts"
def onClickForecasts():
    st.session_state.valH = False
    st.session_state.valDV = False
    st.session_state.valFV = True
    st.session_state.valR = False


# Met à jour les valeurs des variables associées aux widget lors d'un click sur le bouton "View Benchmark"
def onClickResults():
    st.session_state.valH = False
    st.session_state.valDV = False
    st.session_state.valFV = False
    st.session_state.valR = True


# Fonction qui va exécuter le streamlit
def mainApp(pages, dataDict, Product_ids, Store_ids):
    # On initialise les variables associées aux widgets si elles ne sont pas encore initialisées
    initStates()

    # On récupère les valeurs des différents widgets
    button_home, STORE_ID, PRODUCT_ID, buttonDataView, buttonForecastView, model_choice, button_results = buildSidebar(
        pages, Product_ids, Store_ids)

    # On update les valeurs de ces variables en fonction des nouvelles valeurs des widgets
    changeStates(STORE_ID, PRODUCT_ID, model_choice, pages, dataDict)

    # On update les valeurs de ces variables en fonction de ce sur quoi on a cliqué
    if button_home:
        onClickHome()
    if buttonDataView:
        onClickData()
    if buttonForecastView:
        onClickForecasts()
    if button_results:
        onClickResults()

    # On gère l'affichage des pages
    if st.session_state.valH:
        Intro.app()
    if st.session_state.valDV:
        View_data.app(st.session_state.valDF, st.session_state.valP, st.session_state.valS)
    if st.session_state.valFV and st.session_state.valmc != 'None':
        st.session_state.valM.app(st.session_state.valDF)
    if st.session_state.valFV and st.session_state.valmc == 'None':
        st.markdown('<h2><center>You must choose a model to see the forecasts</center></h2>', unsafe_allow_html=True)
        st.image('Pictures/errorSymbol.jpg')
    if st.session_state.valR:
        View_benchmark.app()


# Liste des magasins et produits
Store_ids = ['09', '36']
Product_ids = ['2008', '183', '151', '101', '230']
# Enregistrement des DataFrames
dataDict = storageData(Product_ids, Store_ids)
# Enregistrement des Pages
pages = buildPages()

# Run
mainApp(pages, dataDict, Product_ids, Store_ids)
