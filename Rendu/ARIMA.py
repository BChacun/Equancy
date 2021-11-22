import streamlit as st
import pandas as pd
import numpy as np


def app(df):
    st.title('ARIMA Algorithm')
    st.write('Welcome to app1')
    
    def ARIMA_method(p,d,q):
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

        from statsmodels.tsa.ar_model import AutoReg
        from sklearn.metrics import mean_squared_error
        from math import sqrt



        # split into train and test sets
        X = df.values
        series = df
       

        train, test = X[0:p], X[p:]
        history = [x for x in train]
        


        model = ARIMA(X,order=(p,d,q))
        model_fit = model.fit()
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions = model_fit.predict(start=0, end=len(X)-1, dynamic=False)
        true_predictions = [0]*(len(X))
        for k in range (len(X)):
            if k > p-1:
                true_predictions[k]=predictions[k]
         
             
        for i in range(len(predictions)):
            
                print('predicted=%f, expected=%f' % (predictions[i], X[i]))

        # create results
        results = series
        results["Prediction"] = predictions
        #results["Prediction"] = true_predictions
        
        results.to_csv(os.getcwd() + r'\ARIMA.csv')
        
    try :
        data = pd.read_csv("ARIMA.csv")
    except :
            
        ARIMA_method(30,1,10)
        data = pd.read_csv("ARIMA.csv")

        
            
            
        

    data["weekDate"] = pd.to_datetime(data['weekDate'])
    data.set_index(['weekDate'],inplace=True)
    st.write(data)
        
     # Plots :
    st.line_chart(data=data, width=0, height=0, use_container_width=True)











#streamlit run C:\Users\bapti\Desktop\Equancy\Rendu\main.py
