import warnings
warnings.filterwarnings('ignore')
from Models import Models
import pyodbc
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import datetime
import plotly.express as px
import pickle
from prophet import Prophet

def connect_sqlserver(server='deti-sql-aulas.ua.pt',database='sisqualFORECASTDATA',username='sisqual',password='rh4.0'):
    con = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = con.cursor()

    return con,cursor

def to_supervised(input,n_lags,n_output=1):
    X,y=[],[]
    
    for i in range(len(input)-n_lags):
        aux=[]
        for j in range(i,i+n_lags,1):
            aux.append(input[j])
        if i+n_lags+n_output<len(a):
            X.append(aux)
            y.append(a[i+n_lags:i+n_lags+n_output])

    return np.array(X),np.array(y)

def get_data_tienda(con,tienda):

    query='SELECT*FROM [sisqualFORECASTDATA].[Martim].[ForecastData] Where tienda=?;'
    data=pd.read_sql(query,con,params=[tienda])
    data['fechaHoraInicio']=pd.to_datetime(data['fechaHoraInicio'])
    data['fechaHoraFim']=pd.to_datetime(data['fechaHoraFim'])
    data['fecha']=pd.to_datetime(data['fecha'])
    data['cantidad']=data['cantidad'].astype(float)
    data['numero']=data['numero'].astype(float)
    data['tickets']=data['tickets'].astype(float)
    data['importe']=data['importe'].astype(float)
    data['caja']=data['caja'].astype(int)
    data['tickets']=data['tickets'].apply(lambda x:max(x,0))
    

    return data

def hour_resample(data):

    data=data.drop(columns=['caja'])
    forecast_data=pd.DataFrame(data.resample('H', on='fechaHoraInicio').tickets.sum())
    forecast_data.reset_index(inplace=True)
    
    return forecast_data

def data_augmentation(forecast_data):

    forecast_data['Mês']=forecast_data['fechaHoraInicio'].apply(lambda x: x.month)
    forecast_data['Dia']=forecast_data['fechaHoraInicio'].apply(lambda x: x.weekday())
    forecast_data['Hora']=forecast_data['fechaHoraInicio'].apply(lambda x: x.hour)

    return forecast_data

def to_supervised(a,n_lags,n_output=1):
    X,y=[],[]
    
    for i in range(len(a)-n_lags):
        aux=[]
        for j in range(i,i+n_lags,1):
            aux.append(a[j])
        if i+n_lags+n_output<len(a):
            X.append(aux)
            y.append(a[i+n_lags:i+n_lags+n_output])

    return np.array(X),np.array(y)

def concat(forecast_data,df,lags):

    forecast_data_final=pd.concat([forecast_data.iloc[lags:,:].reset_index(drop=True),df],axis=1,join="inner")
    forecast_data_final=forecast_data_final.drop(columns=['fechaHoraInicio','tickets'])

    return forecast_data_final

def to_supervised2(forecast_data_final,N):

    X=forecast_data_final.iloc[:,:-1].values
    y=forecast_data_final.iloc[:,-1].values
    X_train=X[:-N,:]
    y_train=y[:-N]
    X_test=X[-N:]
    y_test=y[-N:]
    
    return X_train,X_test,y_train,y_test

def train_predict(model_instance,X_train,y_train,X_test,lags,best_metric,cursor,tienda):
    model=model_instance.model
    start=time.time()
    model.fit(X_train,y_train)
    forecast=[]
    for k in range(len(y_test)):
        X_input=list(X_test[k,:3])
        for i in range(lags):
            if(k+i<lags):
                X_input.append(X_train[-1][k+i])
            else:
                X_input.append(forecast[-(lags-k-i)])
        forecast.append(round(max(model.predict(np.array(X_input).reshape((1,len(X_input))))[0],0)))
    
    end=time.time()
    tempo=round(end-start,2)


    sql = "INSERT INTO Martim.models (tienda,model_type,model_pickle,model_metrics,best_metric,model_train_time) VALUES (?,?,?,?,?,?)"
    best_metric,model_metrics=model_instance.model_errors(y_test,forecast,best_metric)
    params = [tienda,model_instance.model_type,pickle.dumps(model),model_metrics,best_metric,tempo]
    cursor.execute(sql, params)
    cursor.commit()

    return forecast,tempo

def train_predict_prophet(model_instance,df_train,df_test,best_metric,cursor,tienda):
    start=time.time()
    m=model_instance.model
    m.add_regressor('add1')
    m.add_regressor('add2')
    m.add_regressor('add3')
    df_train=forecast_data_prophet[:-N]
    df_test=forecast_data_prophet[-N:]
    m.fit(df_train)
    df_pred=m.predict(df_test)
    forecast=df_pred['yhat'].values
    forecast[forecast < 0] = 0

    end=time.time()
    tempo=round(end-start,2)
    


    sql = "INSERT INTO Martim.models (tienda,model_type,model_metrics,best_metric,model_train_time) VALUES (?,?,?,?,?)"
    best_metric,model_metrics=model_instance.model_errors(y_test,forecast,best_metric)
    params = [tienda,model_instance.model_type,model_metrics,best_metric,tempo]
    cursor.execute(sql, params)
    cursor.commit()

    return forecast,tempo




if __name__=='__main__':

    ### params
    
    lags=3
    N=1440
    best_metric='rmse'

    ## Pré-processamento
    con,cursor=connect_sqlserver()
    cursor.execute('DELETE  FROM Martim.models;')
    cursor.commit()

    cursor.execute('SELECT distinct tienda from Martim.ForecastData')
    tiendas_query=cursor.execute('SELECT Distinct tienda from Martim.ForecastData;')
    tiendas=[item[0] for item in tiendas_query]

    for tienda in tiendas:

        data=get_data_tienda(con,tienda)
        forecast_data=hour_resample(data)
        forecast_data=data_augmentation(forecast_data)
        tickets=np.array(forecast_data['tickets'])
        X,y=to_supervised(tickets,lags)
        df=pd.DataFrame(np.hstack((X,y)),columns=['lag'+str(i-1) for i in range(lags+1,0,-1)])
        df=df.rename(columns={'lag0':'target'})
        forecast_data_final=concat(forecast_data,df,lags)
        X_train,X_test,y_train,y_test=to_supervised2(forecast_data_final,N)

        #Modelos
        reg = GradientBoostingRegressor(random_state=0)
        model=Models(reg,'GradientBoosting')
        train_predict(model,X_train,y_train,X_test,forecast_data_final.shape[1]-4,best_metric,cursor,tienda)

        

        forecast_data_prophet=forecast_data.rename(columns={'fechaHoraInicio':'ds','tickets':'y','Mês':'add1','Dia':'add2','Hora':'add3'})
        df_train=forecast_data_prophet[:-N]
        df_test=forecast_data_prophet[-N:]
        m = Prophet()
        model=Models(m,'Prophet')
        train_predict_prophet(model,df_train,df_test,best_metric,cursor,tienda)



    