import sklearn.metrics
import numpy as np
import json

def maape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.arctan(np.abs((y_true-y_pred)/y_true)))

def seasonal_naive(y_train,horizon,s=168):
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

def seasonal_MASE(y_true,y_pred,y_train,s=168):
    
    
    y_pred_naive=seasonal_naive(y_train,len(y_pred),s)
    
    return sklearn.metrics.mean_absolute_error(y_true,y_pred)/sklearn.metrics.mean_absolute_error(y_true,y_pred_naive)



class Models:

    def __init__(self,model,model_type):
        self.model_type=model_type
        self.model=model

    def model_errors(self,y_train,y_true,y_pred,best_metric,index_aberto):

        
        model_metrics={}
        y_pred=np.array(y_pred)[index_aberto]
        y_true=np.array(y_true)[index_aberto]
        model_metrics['y_true']=json.dumps(list(y_true))
        model_metrics['y_pred']=json.dumps(list(y_pred))
        model_metrics['r2_score']=sklearn.metrics.r2_score(y_true,y_pred)
        model_metrics['mse']=sklearn.metrics.mean_squared_error(y_true,y_pred)
        model_metrics['mae']=sklearn.metrics.mean_absolute_error(y_true,y_pred)
        model_metrics['rmse']=np.sqrt(model_metrics['mse'])
        model_metrics['sMASE']=seasonal_MASE(y_true,y_pred,y_train)

        print(self.model_type,model_metrics)
        return round(model_metrics[best_metric],3),json.dumps(model_metrics)

    def model_errors2(self,y_true,y_pred,best_metric,index_aberto):

        
        model_metrics={}
        y_pred=np.array(y_pred)[index_aberto]
        y_true=np.array(y_true)[index_aberto]
        model_metrics['y_true']=json.dumps(list(y_true))
        model_metrics['y_pred']=json.dumps(list(y_pred))
        model_metrics['r2_score']=sklearn.metrics.r2_score(y_true,y_pred)
        model_metrics['mse']=sklearn.metrics.mean_squared_error(y_true,y_pred)
        model_metrics['mae']=sklearn.metrics.mean_absolute_error(y_true,y_pred)
        model_metrics['rmse']=np.sqrt(model_metrics['mse'])
        print(self.model_type,model_metrics)
        
        return round(model_metrics[best_metric],3),json.dumps(model_metrics)
    
    
