import pystan
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
from Models import Models
import pyodbc
import pandas as pd
import time
import numpy as np
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import datetime
import xgboost as xgb
import plotly.express as px
import pickle
import plotly.io as pio
from sklearn.model_selection import TimeSeriesSplit
import json

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

def data_augmentation(forecast_data):

    forecast_data['Mês']=forecast_data['fechaHoraInicio'].apply(lambda x: x.month)
    forecast_data['Dia']=forecast_data['fechaHoraInicio'].apply(lambda x: x.weekday())
    forecast_data['Hora']=forecast_data['fechaHoraInicio'].apply(lambda x: x.hour)
    forecast_data['Fechado']=forecast_data['tickets'].apply(lambda x: fechado(x))
    inicio_2020=datetime.datetime(2020, 1, 1)
    forecast_data=forecast_data[forecast_data['fechaHoraInicio']<inicio_2020]

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

def fechado(tickets):
    ret=None
    
    if tickets==0:
        ret=1
    else:
        ret=0
    return ret

def get_index_aberto(X_test):
    fechado_array=X_test[:,3]
    index_list=[]
    
    for i in range(len(fechado_array)):
        if(fechado_array[i]==0):
            index_list.append(i)
    return index_list

def get_index_aberto_prophet(df_test):
    fechado_array=df_test.values[:,-1]
    index_list=[]
    
    for i in range(len(fechado_array)):
        if(fechado_array[i]==0):
            index_list.append(i)
    return index_list



def train_predict(model_instance,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tienda):
    model=model_instance.model
    start=time.time()
    forecast=np.zeros(len(y_test))
    n=X_train.shape[0]
    sample_weight=[]
    alpha=0.999

    for i in range(n):
        valor=alpha**i
    
        if valor<0.1:
            valor=0.1
        sample_weight.append(valor)
    sample_weight=list(reversed(sample_weight))

    model.fit(X_train,y_train,sample_weight)

    for k in range(len(y_test)):
        X_input=list(X_test[k,:4])
        for i in range(lags):
            if(k+i<lags):
                X_input.append(X_train[-1][k+i])
            else:
                X_input.append(forecast[k+i-lags])
        forecast[k]=round(max(model.predict(np.array(X_input).reshape((1,len(X_input))))[0],0))
    
    end=time.time()
    tempo=round(end-start,2)



    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
    index_aberto=get_index_aberto(X_test)
    best_metric,model_metrics=model_instance.model_errors(y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,json.dumps(model.get_params()),pickle.dumps(model),model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()



    

def save_fig(y_test,forecast,tienda,dir):
    df_plot=pd.DataFrame.from_dict({'x':[i for i in range(len(forecast))],'Previsão':forecast,'Real':y_test})

    fig = px.line(df_plot[720:],x='x',y=df_plot.columns[1:],title='Previsões loja '+str(tienda), labels={
                     "x": "Hora",
                     "value": "#tickets"
                 })
    pio.write_image(fig, dir)



def train_predict_prophet(model_instance,df_train,df_test,best_metric,cursor,tienda):
    start=time.time()
    n=df.shape[0]
    m=model_instance.model
    m.add_regressor('add1')
    m.add_regressor('add2')
    m.add_regressor('add3')
    m.add_regressor('add4')
    m.fit(df_train)
    future = df_test.drop('y', 1)
    forecast=m.predict(future)['yhat'].values
    forecast=np.round(forecast)
    y_test=df_test['y'].values
    index_aberto=get_index_aberto_prophet(df_test)

    end=time.time()
    tempo=round(end-start,2)
    


    sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?)"
    best_metric,model_metrics=model_instance.model_errors(y_test,forecast,best_metric,index_aberto)
    params = [datetime.datetime.now(),tienda,model_instance.model_type,pickle.dumps(m),model_metrics,best_metric,tempo,n]
    cursor.execute(sql, params)
    cursor.commit()

    




if __name__=='__main__':

    ### params
    
    lags=5
    N=1440
    best_metric='r2_score'
    cv=TimeSeriesSplit(n_splits=3)
    popsize=15
    generationsize=3

    ## Pré-processamento
    con,cursor=connect_sqlserver()
    #cursor.execute('DELETE  FROM Martim.models;')
    #cursor.commit()

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

        if forecast_data.shape[0]>0:

            tickets=np.array(forecast_data['tickets'])
            X,y=to_supervised(tickets,lags)
            df=pd.DataFrame(np.hstack((X,y)),columns=['lag'+str(i-1) for i in range(lags+1,0,-1)])
            df=df.rename(columns={'lag0':'target'})
            forecast_data_final=concat(forecast_data,df,lags)
            X_train,X_test,y_train,y_test=to_supervised2(forecast_data_final,N)

        if(forecast_data.shape[0]>0 and X_train.shape[0]>0):
            #Modelos
            forecast_data_prophet=forecast_data.rename(columns={'fechaHoraInicio':'ds','tickets':'y','Mês':'add1','Dia':'add2','Hora':'add3','Fechado':'add4'})
            df_train=forecast_data_prophet[:-N]
            df_test=forecast_data_prophet[-N:]
            m1 = Prophet(daily_seasonality=True,weekly_seasonality=False)
            model_daily=Models(m1,'Prophet_daily')
            train_predict_prophet(model_daily,df_train,df_test,best_metric,cursor,tiendas[i])

            m2=Prophet(weekly_seasonality=True,daily_seasonality=True)
            model_weekly=Models(m2,'Prophet_weekly')
            train_predict_prophet(model_weekly,df_train,df_test,best_metric,cursor,tiendas[i])

            
            m3=Prophet(weekly_seasonality=True,daily_seasonality=True)
            m3.add_seasonality(name='monthly',period=30.5, fourier_order=5)
            model_monthly=Models(m3,'Prophet_monthly')
            train_predict_prophet(model_monthly,df_train,df_test,best_metric,cursor,tiendas[i])


            xgb_reg = xgb.XGBRegressor()
            con,cursor=connect_sqlserver()
            model_xgb=Models(xgb_reg,'XGBoost')
            train_predict(model_xgb,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])
            
            param_grid={'learning_rate':Continuous(0.01,0.9),
            'max_iter':Integer(50,500),'max_leaf_nodes':Integer(5,100),'max_depth':Integer(3,50),
            'max_bins':Integer(5,255),'l2_regularization':Continuous(0.0001,1000)}
            hgbr = HistGradientBoostingRegressor(loss='squared_error')
            hgbr_ga=GASearchCV(estimator=hgbr,
                                    cv=cv,
                                    scoring='neg_mean_squared_error',
                                    population_size=popsize,
                                    generations=generationsize,
                                    tournament_size=2,
                                    elitism=True,
                                    crossover_probability=0.8,
                                    mutation_probability=0.1,
                                    param_grid=param_grid,
                                    criteria='max',
                                    algorithm='eaMuPlusLambda',
                                    n_jobs=-1,
                                    verbose=True)
            hgbr_ga.fit(X_train,y_train)
            hgbr_opt=HistGradientBoostingRegressor()
            hgbr_opt.set_params(**hgbr_ga.best_params_)
            con,cursor=connect_sqlserver()
            model1=Models(hgbr_opt,'HistGradientBoosting')
            train_predict(model1,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])

            param_grid={'n_estimators':Integer(10,200),'learning_rate':Continuous(0,0.6),'loss':Categorical(['linear', 'square', 'exponential'])}
            
            adareg=AdaBoostRegressor()

            adareg_ga=GASearchCV(estimator=adareg,
                        cv=cv,
                        scoring='neg_mean_squared_error',
                        population_size=popsize,
                        generations=generationsize,
                        tournament_size=2,
                        elitism=True,
                        crossover_probability=0.8,
                        mutation_probability=0.1,
                        param_grid=param_grid,
                        criteria='max',
                        algorithm='eaMuPlusLambda',
                        n_jobs=-1,
                        verbose=True)

            adareg_ga.fit(X_train,y_train)
            adareg_opt=AdaBoostRegressor()
            adareg_opt.set_params(**adareg_ga.best_params_)
            con,cursor=connect_sqlserver()
            model2=Models(adareg_opt,'AdaBoost')
            train_predict(model2,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])

            etr=ExtraTreesRegressor(n_jobs=-1)
            con,cursor=connect_sqlserver()
            model3=Models(etr,'ExtraTrees')
            train_predict(model3,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])



            rfr_opt=RandomForestRegressor(bootstrap=True ,max_features=0.8, min_samples_leaf=1, min_samples_split=13, n_estimators=100)
            con,cursor=connect_sqlserver()
            model4=Models(rfr_opt,'RandomForest')
            train_predict(model4,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])

            param_grid={'splitter':Categorical(['best','random']),'criterion':Categorical(['poisson','friedman_mse','squared_error', 'absolute_error']),
            'max_depth':Integer(2,10),'min_samples_split':Integer(2,10),'max_features':Continuous(0,1),
            'max_leaf_nodes':Integer(2,100),'min_impurity_decrease':Continuous(0,0.5),'ccp_alpha':Continuous(0,1)}

            dtr=DecisionTreeRegressor()

            dtr_ga=GASearchCV(estimator=dtr,
                        cv=cv,
                        scoring='neg_mean_squared_error',
                        population_size=popsize,
                        generations=generationsize,
                        tournament_size=2,
                        elitism=True,
                        crossover_probability=0.8,
                        mutation_probability=0.1,
                        param_grid=param_grid,
                        criteria='max',
                        algorithm='eaMuPlusLambda',
                        n_jobs=-1,
                        verbose=True)

            dtr_ga.fit(X_train,y_train)
            dtr_opt=DecisionTreeRegressor()
            dtr_opt.set_params(**dtr_ga.best_params_)
            con,cursor=connect_sqlserver()
            model5=Models(dtr_opt,'DecisionTree')
            train_predict(model5,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])


            gbr=GradientBoostingRegressor()
            con,cursor=connect_sqlserver()
            model6=Models(gbr,'GradientBoosting')
            train_predict(model6,X_train,y_train,X_test,y_test,lags,best_metric,cursor,tiendas[i])



            #save_fig(y_test,forecast,tienda,'./Plots/Previsoes_loja'+str(tienda)+'.png')

        

            #save_fig(y_test,forecast,tienda,'./Plots/Previsoes_loja'+str(tienda)+'.png')



    