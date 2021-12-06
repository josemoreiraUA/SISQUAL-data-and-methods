import sklearn.metrics
import numpy as np

def maape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.arctan(np.abs((y_true-y_pred)/y_true)))

class Models:

    def __init__(self,model,model_type):
        self.model_type=model_type
        self.model=model

    def model_errors(self,y_true,y_pred,best_metric):

        model_metrics={}
        model_metrics['r2_score']=sklearn.metrics.r2_score(y_true,y_pred)
        model_metrics['mse']=sklearn.metrics.mean_squared_error(y_true,y_pred)
        model_metrics['mae']=sklearn.metrics.mean_absolute_error(y_true,y_pred)
        model_metrics['rmse']=np.sqrt(model_metrics['mse'])
        model_metrics['maape']=maape(y_true,y_pred)

        return model_metrics[best_metric],str(model_metrics)
    
    
