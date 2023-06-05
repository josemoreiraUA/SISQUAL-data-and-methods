"""
	Custom Auxiliary Functions

	This file is a module providing custom auxiliary functions.

	Uses 'numpy' and 'scikit-learn'.

	Contains the following functions:

		* closed()
			checks if a store is opened or closed depending on the number of visitors.

		* streming_outlier_detection()
			detects and removes outliers using a streaming z-score outlier detection.

		* model_metrics()
			computes error metrics.

		* to_supervised()
			...
"""

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

def closed(num_visitors: int) -> {0, 1}:
    """
	Decides if a store is opened or closed based on the number of visitor.

    If there are no visitors => store is closed (almost surely).
	Ideally, it would be better to have a timetable including all the schedule
	(SISQUAL probably knows this information).
	
    Parameters
    ----------
    num_visitors : int
        The number of visitors.

    Raises
    ------
    Exception
        If the num_visitors < 0.
			
    Returns
    -------
    A value in {0, 1}
        0 -> store is opened.
        1 -> store is closed.
    """

    if num_visitors < 0:
        raise Exception('num_visitors must be >= 0. Its value was: {}'.format(num_visitors))

    # Store is closed.
    if num_visitors == 0:
        return 1

    # Store is open.
    return 0

def streming_outlier_detection(ts, k, s, alpha):
    """
	streaming z-score outlier detection.

    Parameters
    ----------
		ts : 
			time series values
		k : 	
			last k seasonal elements for comparison
		s : int 	
			seasonality period
		alpha : float
			standard deviation multiplication factor 

	Returns
    ----------
		outliers : list
		    index and values
		ts_copy	
		    new timeseries with imputation on the outliers	
	"""

    ts_copy=np.copy(ts)
    outliers=[]

    prev_elements=np.ones(k)

    for i in range(k*s,len(ts)):
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
            
    return outliers, ts_copy

def model_metrics(y_true, y_pred):
    """
	Computes some error metrics (MASE or RelMAE are better choices).

	Metrics computed:
	
		* rmse
		* mae
		* r2_score
    """

    d_metrics = {}
    d_metrics['rmse'] = np.sqrt(mean_squared_error(y_true,y_pred))
    d_metrics['mae'] = mean_absolute_error(y_true,y_pred)
    d_metrics['r2_score'] = r2_score(y_true,y_pred)
    
    return d_metrics

def to_supervised(timeseries, n_lags: int, n_output: int = 1):
    """
	Class method docstrings go here.
	
	Example usage:
    ----------
		timeseries = [1,2,3,4,5,6,7,8,9,10]
		n_lags = 5
		n_output = 1
		
		[1,2,3,4,5] 6
		[2,3,4,5,6] 7
		[3,4,5,6,7] 8
		[4,5,6,7,8] 9
		[5,6,7,8,9] 10	
    """

    #from app.core.config import settings
    from app.models.models import MyException

    #settings.file_logger.info(str(n_lags))
    #settings.file_logger.info(str(n_output))

    N = len(timeseries)

    if N - n_lags - n_output + 1 <= 1:
        raise MyException(f'Training data size ({N} hours) too small (must be at least {n_lags + n_output + 1} hours)')

    #settings.file_logger.info(str(N))
    #settings.file_logger.info(str(N - n_lags - n_output + 1))

    X = np.zeros((N - n_lags - n_output + 1, n_lags))
    y = np.zeros((X.shape[0], n_output))

    #settings.file_logger.info(X)
    #settings.file_logger.info(str(X.shape))
    #settings.file_logger.info(y)
    #settings.file_logger.info(str(y.shape))

    #h = 0
    #g = 0
    #f = 0

    for i in range(N - n_lags):
        if i + n_lags + n_output > N:
            #g += 1
            break

        aux = np.zeros(n_lags)

        for j in range(i, i + n_lags, 1):
            aux[j - i] = timeseries[j]

        #if i + n_lags + n_output <= N:
        X[i,:] = aux
        y[i,:] = timeseries[i + n_lags : i + n_lags + n_output]
        #h += 1

        #f += 1

    #settings.file_logger.info(str(h))
    #settings.file_logger.info(str(g))
    #settings.file_logger.info(str(f))

    return X, y

def to_supervised_multivariate(y1, y2, n_lags: int, n_output: int = 1):
    N = len(y1)

    n_rows = N - n_lags - n_output + 1
    
    X = np.zeros((n_rows, n_lags*2))
    y = np.zeros((n_rows, n_output*2))
    
    for i in range(N - n_lags):
        if i + n_lags + n_output > N:
            break

        #aux = np.zeros(n_lags*2)
        aux = []

        for j in range(i, i + n_lags, 1):
            #aux[j - i] = y1[j]
            aux += [y1[j]]
            aux += [y2[j]]

        X[i,:] = aux
        
        aux = []

        for j in range(i + n_lags, i + n_lags + n_output, 1):
            aux += [y1[j]]
            aux += [y2[j]]
        
        y[i,:] = aux
        #y[i,:] = timeseries[i + n_lags : i + n_lags + n_output]

    return X, y

if (__name__ == '__main__'):
    print('Module containing custom auxiliary functions.')
