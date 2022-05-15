import warnings
warnings.filterwarnings('ignore')
import pystan
from prophet import Prophet
from Models import Models
import pyodbc
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import datetime
import json
import pickle
from tbats import TBATS
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,Nadam,RMSprop,Adamax
from tensorflow.keras.layers import Masking
from sklearn.preprocessing import MinMaxScaler




def connect_sqlserver(server='deti-sql-aulas.ua.pt',database='sisqualFORECASTDATA',username='sisqual',password='rh4.0'):
    con = pyodbc.connect('DRIVER={SQL Server Native Client 11.0};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = con.cursor()

    return con,cursor

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

def fechado(tickets):
    ret=None
    
    if tickets==0:
        ret=1
    else:
        ret=0

    return ret

def data_augmentation(forecast_data):

    forecast_data['Mês']=forecast_data['fechaHoraInicio'].apply(lambda x: x.month)
    forecast_data['Dia']=forecast_data['fechaHoraInicio'].apply(lambda x: x.weekday())
    forecast_data['Hora']=forecast_data['fechaHoraInicio'].apply(lambda x: x.hour)
    forecast_data['Fechado']=forecast_data['tickets'].apply(lambda x: fechado(x))
    t=datetime.datetime(2019, 12, 1)
    forecast_data=forecast_data[forecast_data['fechaHoraInicio']<t]

    return forecast_data

def streming_outlier_detection(ts,k,s,N,alpha):
    
    #streaming z-score outlier detection
    
    #input
    # ts - time series values
    # k - last k seasonal elements for comparison
    # s - seasonality period
    # alpha - standard deviation multiplication factor 
    # N - forecast horizon
    
    #output
    #outliers list: index and values
    # ts_copy: new timeseries with imputation on the outliers
    
    ts_copy=np.copy(ts)
    outliers=[]
    
    prev_elements=np.ones(k)
    
    for i in range(k*s,len(ts)-N):
        
        for j in range(k):
            prev_elements[j]=ts[i-(j+1)*s]
            
        mu=np.mean(prev_elements)
        sigma=np.std(prev_elements)
        
        upper_whisker=mu+alpha*sigma
        lower_whisker=mu-alpha*sigma
        
        if ts[i]>upper_whisker or ts[i]<lower_whisker:
            outliers.append([i,ts[i]])
            masked=np.ma.masked_equal(prev_elements, 0)
            ts_copy[i]=np.where(masked.mean()==None, 0, masked.mean())
            
    return outliers,ts_copy

def seasonal_naive(y_train,horizon,s):
    # input
    
    # y_train - input time series
    # forecast - horizon
    # s - seasonal period
    
    #output
    # y_pred - forecast values
    
    y_pred=np.zeros(horizon)
    N=len(y_train)
    
    for i in range(horizon):
        if i<s:
            y_pred[i]=y_train[N-s+i]
        else:
            y_pred[i]=y_pred[i-s]
    
    return y_pred

def train_predict_1(model_instance,X_train,y_train,X_test,y_test,best_metric,cursor,tienda):
    model=model_instance.model
    n=X_train.shape[0]

    start=time.time()
    model.fit(X_train,y_train)
    forecast=model.predict(X_test)
    forecast[forecast<0]=0
    forecast=np.around(forecast)
    end=time.time()
    tempo=round(end-start,2)

    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,json.dumps(model.get_params()),None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()

def train_predict_2(model_instance,y_train,y_test,horizon,s,best_metric,cursor,tienda):

    start=time.time()
    n=len(y_train)
    forecast=seasonal_naive(y_train,horizon,s)
    forecast[forecast<0]=0
    forecast=np.around(forecast)
    end=time.time()
    tempo=round(end-start,2)

    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,None,None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()


def train_predict_3(model_instance,df_proph,nprev,best_metric,cursor,tienda):
    model=model_instance.model
    n=df_proph.shape[0]-nprev

    start=time.time()
    model.fit(df_proph[:-nprev])
    future=model.make_future_dataframe(freq='H',periods=nprev)
    forecast=model.predict(future)['yhat']
    forecast=np.around(forecast)
    forecast[forecast<0]=0

    end=time.time()
    tempo=round(end-start,2)

    y_train=df_proph[:-nprev]['y']
    y_test=df_proph[-nprev:]['y']

    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,None,None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()

def train_predict_4(model_instance,mms_y,X_train,y_train,X_test,y_test,best_metric,cursor,tienda):
    model=model_instance.model
    n=X_train.shape[0]

    start=time.time()
    model.fit(X_train,y_train)
    forecast=model.predict(X_test)
    forecast=mms_y.inverse_transform(forecast).flatten()
    forecast=np.around(forecast)
    forecast[forecast<0]=0
    end=time.time()
    tempo=round(end-start,2)
    y_test=y_test.flatten()

    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    y_train=y_train[-1,:].reshape(1,-1)
    y_train=mms_y.inverse_transform(y_train).flatten()
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,json.dumps(model.get_params()),None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()


def train_predict_5(model_instance,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tienda):
    model=model_instance.model
    start=time.time()
    model.fit(X_train,y_train)
    forecast=np.zeros(len(y_test))
    n=X_train.shape[0]

    for k in range(len(y_test)):
        X_input=list(X_test[k,:4])
        for i in range(lags):
            if(k+i<lags):
                X_input.append(X_train[-1][k+i])
            else:
                X_input.append(forecast[k+i-lags])
        X_input=np.array(X_input)
        forecast[k]=round(max(model.predict(X_input.reshape((1,len(X_input))))[0],0))

    end=time.time()
    tempo=round(end-start,2)
    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,json.dumps(model.get_params()),None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()


def train_predict_6(model_instance,y_train,y_test,best_metric,cursor,tienda,N):
    model=model_instance.model
    start=time.time()
    n=len(y_train)

    # Fit model
    fitted_model = model.fit(y_train)

    # Forecast 720 steps ahead
    forecast = fitted_model.forecast(steps=N)
    forecast=np.around(forecast)

    end=time.time()
    tempo=round(end-start,2)
    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,None,None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()

def train_predict_7(model_instance,X_train,y_train,y_test,steps,best_features,best_metric,cursor,tienda):
    model=model_instance.model
    start=time.time()
    n=X_train.shape[0]
    model.fit(X_train[:,best_features],y_train)
    forecast=np.zeros(len(y_test))

    for k in range(len(y_test)):
        X_input=[]
        for i in range(steps):
            if(k+i<steps):
                X_input.append(X_train[-1][k+i])
            else:
                X_input.append(forecast[-(steps-k-i)])
        X_input=np.array(X_input)[best_features]
        forecast[k]=round(max(model.predict(np.array(X_input).reshape((1,len(X_input))))[0],0))
    
    end=time.time()
    tempo=round(end-start,2)
    
    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,None,None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()


def train_predict_TBATS(model_instance,y_train,y_test,horizon,best_metric,cursor,tienda):
    
    model=model_instance.model
    start=time.time()
    n=len(y_train)
    fitted_model=model.fit(y_train)
    forecast = fitted_model.forecast(steps=horizon)
    forecast=np.around(forecast)
    forecast[forecast<0]=0
    end=time.time()
    tempo=round(end-start,2)

    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,None,None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()



def train_predict_LSTM(model_instance,mms_y,epochs,batch_size,X_train,y_train,X_test,y_test,best_metric,cursor,tienda):
    model=model_instance.model
    n=X_train.shape[0]

    start=time.time()
    model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size)
    forecast=model.predict(X_test)
    forecast=mms_y.inverse_transform(forecast).flatten()
    forecast=np.around(forecast)
    forecast[forecast<0]=0
    end=time.time()
    tempo=round(end-start,2)
    y_test=y_test.flatten()

    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=np.where(y_test != 0)[0]
    y_train=y_train[-1,:].reshape(1,-1)
    y_train=mms_y.inverse_transform(y_train).flatten()
    best_metric,model_metrics=model_instance.model_errors(y_train,y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,None,None,model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()






def to_supervised(timeseries,n_lags,n_output=1):
    
    N=len(timeseries)
    X=np.zeros((N-n_lags-n_output+1,n_lags))
    y=np.zeros((X.shape[0],n_output))
    
    for i in range(N-n_lags):
        aux=np.zeros(n_lags)
        for j in range(i,i+n_lags,1):
            aux[j-i]=timeseries[j]
        if i+n_lags+n_output<=N:
            X[i,:]=aux
            y[i,:]=timeseries[i+n_lags:i+n_lags+n_output]

    return X,y

if __name__=='__main__':

    best_metric='sMASE'
    N=720

    ## Pré-processamento
    con,cursor=connect_sqlserver()

    cursor.execute('SELECT distinct tienda from Martim.ForecastData')
    tiendas_query=cursor.execute('SELECT Distinct tienda from Martim.ForecastData;')
    tiendas=[item[0] for item in tiendas_query]
    max_tienda=cursor.execute('SELECT max(tienda)from Martim.models;')

    for i in max_tienda:
        max_tienda=i[0]

    if max_tienda!=None:
        cursor.execute('DELETE from Martim.models where TIENDA='+str(max_tienda)+';')
        cursor.commit()
        tiendas.sort()
        max_tienda_index=tiendas.index(max_tienda)
    else:
        max_tienda_index=0
    
    for i in range(max_tienda_index,len(tiendas)):
        print('Calculating models for tienda '+str(tiendas[i]))
        data=get_data_tienda(con,tiendas[i])
        forecast_data=hour_resample(data)
        forecast_data=data_augmentation(forecast_data)


        if forecast_data.shape[0]>9072+N:
            outliers,new_ts=streming_outlier_detection(forecast_data['tickets'].values,k=53,s=168,N=N,alpha=2)
            forecast_data['filtered_tickets']=new_ts

            X=forecast_data[['Mês','Dia','Hora','Fechado']]
            X_train=X[:-N].values
            X_test=X[-N:].values
            y_train=forecast_data[:-N]['filtered_tickets'].values
            y_test=forecast_data[-N:]['tickets'].values

            con,cursor=connect_sqlserver()
            model_tbats = TBATS(seasonal_periods=[168,672,8736],use_arma_errors=False,use_box_cox=False)
            model_tbats_=Models(model_tbats,'TBATS')
            train_predict_TBATS(model_tbats_,y_train,y_test,N,best_metric,cursor,tiendas[i])

            con,cursor=connect_sqlserver()
            model_seasonal_naive=Models(None,'DaySeasonalNaive')
            train_predict_2(model_seasonal_naive,y_train,y_test,N,24,best_metric,cursor,tiendas[i])

            con,cursor=connect_sqlserver()
            model_seasonal_naive=Models(None,'WeekSeasonalNaive')
            train_predict_2(model_seasonal_naive,y_train,y_test,N,168,best_metric,cursor,tiendas[i])

            

            con,cursor=connect_sqlserver()
            model_seasonal_naive=Models(None,'YearSeasonalNaive')
            train_predict_2(model_seasonal_naive,y_train,y_test,N,8736,best_metric,cursor,tiendas[i])



            # Create estimator
            #tbats = TBATS(seasonal_periods=[168,672,8736],use_arma_errors=False,use_box_cox=False)
            #model_tbats=Models(tbats,'TBATS')
            #train_predict_6(model_tbats,y_train,y_test,best_metric,cursor,tiendas[i],N)
            
            

            
            proph=Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=False)
            proph.add_seasonality(name='monthly',period=28, fourier_order=5)
            proph.add_seasonality(name='yearly',period=364, fourier_order=5)
            d={'ds':forecast_data['fechaHoraInicio'].values,'y':forecast_data['filtered_tickets'].values}
            df_proph=pd.DataFrame(d)
            con,cursor=connect_sqlserver()
            model_proph=Models(proph,'Prophet')
            train_predict_3(model_proph,df_proph,N,best_metric,cursor,tiendas[i])

        

            hgbr=HistGradientBoostingRegressor()
            con,cursor=connect_sqlserver()
            model_hgbr=Models(hgbr,'Basic_HistGradientBoosting')
            train_predict_1(model_hgbr,X_train,y_train,X_test,y_test,best_metric,cursor,tiendas[i])

            lags=168
            X,y=to_supervised(forecast_data['filtered_tickets'].values,n_lags=lags)
            df=pd.DataFrame(np.hstack((X,y)),columns=['lag'+str(i-1) for i in range(lags+1,0,-1)])
            df=df.rename(columns={'lag0':'target'})
            forecast_data_final=pd.concat([forecast_data.iloc[lags:,:].reset_index(drop=True),df],axis=1,join="inner")
            forecast_data_final=forecast_data_final.drop(columns=['fechaHoraInicio','tickets','filtered_tickets'])
            X=forecast_data_final.iloc[:,:-1].values
            y=forecast_data_final.iloc[:,-1].values
            X_train=X[:-N,:]
            y_train=y[:-N]
            X_test=X[-N:]
            y_test=y[-N:]
            hgbr=HistGradientBoostingRegressor()
            con,cursor=connect_sqlserver()
            model_hgbr_2=Models(hgbr,'Lag_HistGradientBoosting')
            train_predict_5(model_hgbr_2,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])

            lags=8904
            X,y=to_supervised(forecast_data['filtered_tickets'].values,n_lags=lags)
            df=pd.DataFrame(np.hstack((X,y)),columns=['lag'+str(i-1) for i in range(lags+1,0,-1)])
            df=df.rename(columns={'lag0':'target'})
            best_features=[i for i in range(0,lags,168)]
            X=df.iloc[:,:-1].values
            y=df.iloc[:,-1].values
            X_train=X[:-N,:]
            y_train=y[:-N]
            y_test=y[-N:]
            hgbr=HistGradientBoostingRegressor()
            con,cursor=connect_sqlserver()
            model_seasonal_lag=Models(hgbr,'SeasonalLag_HistGradientBoosting')
            train_predict_7(model_seasonal_lag,X_train,y_train,y_test,lags,best_features,best_metric,cursor,tiendas[i])


            

            
            mlpr=MLPRegressor(hidden_layer_sizes=(1080, 720, 360), max_iter=100)
            con,cursor=connect_sqlserver()
            model_mlpr=Models(mlpr,'MLP')
            X,y=to_supervised(forecast_data['filtered_tickets'].values,n_lags=N,n_output=N)
            X_train=X[:-1]
            y_train=y[:-1]
            X_test=X[-1:]
            y_test=y[-1:]
            mms_X=MinMaxScaler()
            mms_y=MinMaxScaler()
            X_train=mms_X.fit_transform(X_train)
            y_train=mms_y.fit_transform(y_train)
            X_test=mms_X.transform(X_test)
            train_predict_4(model_mlpr,mms_y,X_train,y_train,X_test,y_test,best_metric,cursor,tiendas[i])

            X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
            X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
            model = Sequential()
            model.add(Masking(mask_value=0,input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(LSTM(100, activation='relu', dropout=0.05,return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
            model.add(LSTM(100, activation='relu',dropout=0.05))
            model.add(Dense(N))
            opt=Adam(learning_rate=0.0001,clipnorm=1,clipvalue=1)
            model.compile(optimizer=opt, loss='mse')
            con,cursor=connect_sqlserver()
            model_lstm=Models(model,'LSTM')
            train_predict_LSTM(model_lstm,mms_y,10,500,X_train,y_train,X_test,y_test,best_metric,cursor,tiendas[i])

