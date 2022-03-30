import json
from Main import connect_sqlserver
from Models import Models
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import datetime
import pickle
import time
from sklearn.linear_model import RidgeCV

if __name__=='__main__':

    
    cv=TimeSeriesSplit(n_splits=10)
    con,cursor=connect_sqlserver()
    cursor.execute("""DELETE
    from Martim.models
    where model_type='WeightedAverageStacking'""")
    cursor.commit()
    tiendas_query=cursor.execute('SELECT distinct tienda from Martim.models')
    tiendas=[item[0] for item in tiendas_query]

    for tienda in tiendas:
        best_metric='r2_score'
        print('Stacking for tienda: '+str(tienda))

        query="""SELECT top 3 model_metrics
        FROM [sisqualFORECASTDATA].[Martim].[models]
        where tienda=?
        order by best_metric desc"""
        r=cursor.execute(query,tienda)
        lst_stacking=[]
        coef=np.zeros(3)
        r2_scores=np.zeros(3)
        soma=0
        i=0

        for row in r:
            d=json.loads(row.model_metrics)
            y_pred=json.loads(d['y_pred'])
            lst_stacking.append(y_pred)
            y_true=np.array(json.loads(d['y_true']))
            json.loads(d['y_pred'])
            r2_scores[i]=d["r2_score"]
            i+=1
            soma+=d["r2_score"]
        
        coef[0]=r2_scores[0]/soma
        coef[1]=r2_scores[1]/soma
        coef[2]=r2_scores[2]/soma

        

        np_stacking=np.array(lst_stacking).T
        N=round(np_stacking.shape[0]*0.7)
        np_stacking_train=np_stacking[:N,]
        y_true_train=y_true[:N]
        np_stacking_test=np_stacking[N:,]
        y_true_test=y_true[N:]

        start=time.time()
        forecast=np.zeros(len(y_true))

        for i in range(len(forecast)):
            forecast[i]=np.around(np.dot(np_stacking[i,:],coef))

        end=time.time()
        tempo=round(end-start,2)
        model=Models(None,'WeightedAverageStacking')
        con,cursor=connect_sqlserver()
        sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
        index_aberto=[i for i in range(len(y_true))]
        best_metric,model_metrics=model.model_errors2(y_true,forecast,best_metric,index_aberto)
        params = [datetime.datetime.now(),tienda,model.model_type,None,None,model_metrics,best_metric,tempo,len(forecast)]
        cursor.execute(sql, params)
        cursor.commit()
        

       # ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10,100,1000,10000],cv=cv)
        #if np_stacking_train.shape[0]>10:
         #   ridge.fit(np_stacking_train, y_true_train)
          #  forecast=ridge.predict(np_stacking_test)
           # forecast=np.round(forecast)
           # end=time.time()
           # tempo=round(end-start,2)
           # model=Models(ridge,'Ridge_stacking')
           # con,cursor=connect_sqlserver()
           # sql = "INSERT INTO Martim.models (ts,tienda,model_type,model_params,model_pickle,model_metrics,best_metric,model_train_time,train_size) VALUES (?,?,?,?,?,?,?,?,?)"
           # index_aberto=[i for i in range(len(y_true_test))]
           # best_metric,model_metrics=model.model_errors(y_true_test,forecast,best_metric,index_aberto)
           # params = [datetime.datetime.now(),tienda,model.model_type,str(ridge.get_params()),pickle.dumps(ridge),model_metrics,best_metric,tempo,N]
           # cursor.execute(sql, params)
           # cursor.commit()