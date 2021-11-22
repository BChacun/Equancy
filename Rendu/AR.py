import streamlit as st
import pandas as pd
import numpy as np


def app(df):
    st.title('AR Algorithm')
    st.write('Welcome to app1')
    
    
    
    
    
    
   
    def AR(p):

        
        # IMPORTS :
        import os
        import numpy as np
        import pandas as pd
        from pandas import datetime
        from pandas import read_csv
        from pandas import DataFrame
        from statsmodels.tsa.arima.model import ARIMA
        from pandas.plotting import lag_plot
        from pandas import concat
        from pandas.plotting import autocorrelation_plot
        
        # AR (p):
        # paramètre p : l'AR prend en compte les p valeurs précédentes pour calculer la prédiction
        
        
    
        # create and evaluate a static autoregressive model
        from statsmodels.tsa.ar_model import AutoReg
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        
        # split dataset
        X = df.values
        train, test = X[0:p], X[p:]
        # train autoregression
        model = AutoReg(X,p)
        model_fit = model.fit()
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions = model_fit.predict(start=0, end=len(X)-1, dynamic=False)
        for i in range(len(predictions)):
        	print('predicted=%f, expected=%f' % (predictions[i], X[i]))
        rmse = sqrt(mean_squared_error(test, predictions[p:]))
        print('Test RMSE: %.3f' % rmse)
        
        
        
        # create results
        results = df
        results["Prediction"] = predictions
        
        results.to_csv(os.getcwd() + r'\AR.csv')
        
    try :
        data = pd.read_csv("AR.csv")
    except :
        AR(50)
        data = pd.read_csv("AR.csv")
        

    data["weekDate"] = pd.to_datetime(data['weekDate'])
    data.set_index(['weekDate'],inplace=True)
    st.write(data)
    
    # Plots :
    st.line_chart(data=data, width=0, height=0, use_container_width=True)









# streamlit run C:\Users\bapti\Desktop\Equancy\Rendu\main.py
