from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from src.visualization import data_visualization as dtview
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score


# %% md

## Funtion for outlier detection

# %%

def detect_outliers(dataframe, detect_method, replace_prophet=False, prophet_interval_width=0.95,
                    outliers_fraction=0.05):
    '''
    replace_prophet -> apenas para desenrascar!!
    Enter a time dataframe with the following columns: {ds: timeline ; y: values}
    Enter one of the following outliers detection methods: {prophet; svm; isoforest; gaussian; kmeans}
    '''

    if detect_method == 'prophet':
        print('Using Prophet to detect anomalies! Confidence interval of', (prophet_interval_width * 100), '%.\n')
        # fit the data
        model = Prophet(interval_width=prophet_interval_width)
        model = model.fit(dataframe)
        df_anom = model.predict(dataframe)
        df_anom['y'] = dataframe['y'].reset_index(drop=True)

        # flag anomalies
        df_anom['anomaly'] = 0
        df_anom.loc[df_anom['y'] > df_anom['yhat_upper'], 'anomaly'] = 1
        df_anom.loc[df_anom['y'] < df_anom['yhat_lower'], 'anomaly'] = 1

        # provisório: substituir outliers com valores previstos do prophet
        if replace_prophet:
            df_anom['new_y'] = df_anom['y']
            df_anom.loc[df_anom['anomaly'] != 0, 'new_y'] = df_anom['yhat']

    elif detect_method == 'svm':
        print('Using a one-class SVM to detect anomalies! Outlier fraction of', (outliers_fraction * 100), '%.\n')
        # normalize the y column
        df_anom = dataframe.copy()
        data = df_anom[['y']]
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # fit the data
        model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
        model.fit(data)
        df_anom['anomaly'] = pd.Series(model.predict(data))

        # flag anomalies
        df_anom.loc[df_anom['anomaly'] == 1, 'anomaly'] = 0
        df_anom.loc[df_anom['anomaly'] == -1, 'anomaly'] = 1

    elif detect_method == 'isoforest':
        print('Using an Isolation Forest to detect anomalies! Outlier fraction of', (outliers_fraction * 100), '%.\n')
        # normalize the y column
        df_anom = dataframe.copy()
        data = df_anom[['y']]
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # fit the data
        model = IsolationForest(contamination=outliers_fraction)
        model.fit(data)
        df_anom['anomaly'] = pd.Series(model.predict(data))

        # flag anomalies
        df_anom.loc[df_anom['anomaly'] == 1, 'anomaly'] = 0
        df_anom.loc[df_anom['anomaly'] == -1, 'anomaly'] = 1

    elif detect_method == 'gaussian':
        print('Using the Gaussian Distribution to detect anomalies! Outlier fraction of', (outliers_fraction * 100),
              '%.\n')
        # arrange the data
        df_anom = dataframe.copy()
        data = dataframe[['y']]
        x = data.values.reshape(-1, 1)

        # fit the data
        model = EllipticEnvelope(contamination=outliers_fraction)
        model.fit(x)
        df_anom['deviation'] = model.decision_function(x)
        df_anom['anomaly'] = model.predict(x)

        # flag anomalies
        df_anom.loc[df_anom['anomaly'] == 1, 'anomaly'] = 0
        df_anom.loc[df_anom['anomaly'] == -1, 'anomaly'] = 1

    elif detect_method == 'kmeans':
        print('Using the Gaussian Distribution to detect anomalies! Outlier fraction of', (outliers_fraction * 100),
              '%.\n')
        '''
        inserir código para detetar n_clus automaticamente
        '''
        n_clus = 3

        # normalize the data
        df_anom = dataframe.copy()
        scaler = StandardScaler()
        data = scaler.fit_transform(df_anom[['y']])

        # fit the data
        model = KMeans(init='random', n_clusters=n_clus, n_init=10, max_iter=200)
        model.fit(data)
        print('The lowest SSE value:', model.inertia_, '\n\nFinal locations of the centroid:', model.cluster_centers_,
              '\n\nThe number of iterations required to converge:', model.n_iter_)

        # find distance to center for each point
        distance = pd.Series()
        for i in range(0, len(data)):
            Xa = np.array(data[i])
            Xb = model.cluster_centers_[model.labels_[i] - 1]
            distance.at[i] = np.linalg.norm(Xa - Xb)

        # flag anomalies
        n_outliers = int(outliers_fraction * len(distance))
        threshold = distance.nlargest(n_outliers).min()
        df_anom['anomaly'] = (distance >= threshold).astype(int)
        df_anom['k_label'] = model.labels_

    # count the number of anomalies flagged
    num_anom = np.count_nonzero(df_anom['anomaly'].to_numpy())
    print('A total of', num_anom, 'anomalies were flagged!\n')
    dtview.plot_outliers(dataframe, df_anom)

    return df_anom
