# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import itertools
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from src.data import datacleaner as dtclean, dataframegenerator as gendf
from src.data import make_dataset as make_ds, positivize_model as pm
from src.data import detect_outliers as dout
from src.visualization import data_visualization as dtview
import src.forecast_metrics as fm
import locale
# import scipy.optimize as optim
import random
#from alibi_detect.od import OutlierProphet
#from alibi_detect.utils.fetching import fetch_detector
#from alibi_detect.utils.saving import save_detector, load_detector
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import tqdm
from itertools import repeat
from multiprocessing import Process, Manager
from multiprocessing import Pool
import timeit

# time series analysis

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import settings as settings

'''
@author: Clony Abreu

this python script prepares the data to be explored by fbprophet.
'''
path, csvpath, imgpath, neg_values_path, large_file_path, model_path = settings.get_file_path()


#def detect_prophet_outlier(df_Train, df_Test):
    #     # initialize, fit and save outlier detector
#    od = OutlierProphet(threshold=.8)
#    od.fit(df_Train)
#    save_detector(od, path+'/models/outlierdetect_')
#    od_preds = od.predict(
#        df_Test,
#        return_instance_score=True,
#        return_forecast=True
#    )
#    return od, od_preds

#format = "%Y-%m-%d %H:%M:%S"


#def detect_outlier(data_1):
#    """Função para identificar os dados outliers de um dataframe, baseado no z-score"""
#    outliers = []
#    outliers_moments = []
#    threshold = 3
#    mean_1 = np.mean(data_1)
#    std_1 = np.std(data_1)
#    for y in data_1:
#        z_score = (y - mean_1) / std_1
#        if np.abs(z_score) > threshold:
#            outliers.append(y)
#    return outliers


def get_percentile(df, percentile_rank):

    # First, sort by ascending number of sales or number of clients, reset the indices
    try:
        df = df.sort_values(by='y').reset_index()
    except ValueError as err:
        print('no need to reset index ', err)

    # Rule of three to get the y column index
    index = (len(df.index)-1) * percentile_rank / 100.0
    index = int(index)

    # Return the y corresponding to the percentile rank
    return df.at[index, 'y']

def compute_iqr(df):
    q3 = get_percentile(df, 75) # 75%
    q1 = get_percentile(df, 25) # 25%
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return lower_fence, upper_fence


def calculate_iqr_score(df):
    """Calcula o iqr_score de um dataset ordenado"""
    df = df.sort_values(by=['y'])
    sorted_dataset = df['y'].to_list
    q1, q3 = np.percentile(sorted_dataset, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return lower_fence, upper_fence


def hr_func(ts):
    """ Função que retorna a hora de um objeto timestamp."""
    return ts.hour


def func_logistic(t, a, b, c):
    """Função que define os coeficientes para estimar."""
    return c / (1 + a * np.exp(-b * t))


def find_daily_maxes(x, is_serie=False):
    """Retorna a medida maxima de cada dia e quando ocorreu em um dataframe. O objeto retornado é um dataframe"""
    if is_serie:
        x = x.copy().to_frame()
    else:
        x = x.groupby('y').max()
    result = pd.concat([x.groupby('y').max(),
                        x.groupby('y').idxmax()], axis=1).iloc[:, [0, 1]]
    result.columns = ['date', 'value']
    return result.set_index('date')


def calc_z_score(df):
    """Calcula o z_score do dataframe"""
    z = np.abs(stats.zscore(df))
    return z


def tuning_model_refactored(df, inicio=None, fim=None, periodo=None, frequencia=None,
                            interval_width=[0.98, 0.95, 0.90],
                            cap=None,
                            floor=None,
                            loja=None,
                            tipo=None,
                            changepoint_prior_scale=[15, 20, 25],
                            seasonality_prior_scale=[15, 20, 25],
                            holidays_prior_scale=[15, 20, 25],
                            n_changepoints=[30, 50, 66],
                            fourier_order=[10, 20, 30]):
    df_treino, df_teste = configura_dataframe_treino_teste(df, inicio=inicio, fim=fim)
    df_original = df
    """ Realiza testes com os parâmetros especificados para o modelo, a fim de identificar combinação que obtém
        o melhor mape"""
    nome_arquivo_csv = 'model_parameters_' + loja + '_' + tipo + '.csv'
    params_grid = {'seasonality_mode': ('multiplicative', 'additive'),
                   'changepoint_prior_scale': changepoint_prior_scale,
                   'seasonality_prior_scale': seasonality_prior_scale,
                   'holidays_prior_scale': holidays_prior_scale,
                   'n_changepoints': n_changepoints,
                   'fourier_order': fourier_order,
                   'interval_width': interval_width}
    strt = '2020-05-04 08:30:00'
    end = '2020-05-26 21:30:00'
    grid = ParameterGrid(params_grid)
    prophet_parameters = pd.DataFrame(columns=['MAPE', 'MAAPE', 'seasonality_mode',
                                               'changepoint_prior_scale',
                                               'holidays_prior_scale',
                                               'n_changepoints',
                                               'fourier_order',
                                               'interval_width'])
    for p in grid:
        test = pd.DataFrame()
        print(p)
        random.seed(0)
        train_model = Prophet(changepoint_prior_scale=p['changepoint_prior_scale'],
                              holidays_prior_scale=p['holidays_prior_scale'],
                              n_changepoints=p['n_changepoints'],
                              seasonality_mode=p['seasonality_mode'],
                              weekly_seasonality=True,
                              daily_seasonality=True,
                              yearly_seasonality=False,
                              holidays=dtclean.get_Holiday(),
                              fourier_order=p['fourier_order'],
                              interval_width=p['interval_width'])
        train_model.add_country_holidays(country_name='BR')
        train_model.fit(df_treino)
        train_forecast = train_model.make_future_dataframe(periods=periodo, freq=frequencia, include_history=False)
        train_forecast = train_model.predict(train_forecast)
        test = train_forecast[['ds', 'yhat']]
        Actual = df_original[(df_original['ds'] > strt) & (df_original['ds'] <= end)]
        MAPE = fm.mape(Actual['y'], test['yhat'])
        MAAPE = fm.maape(Actual['y'], test['yhat'])
        print('Mean Absolute Percentage Error(MAPE)------------------------------------', MAPE)
        print('Mean Arctangent Absolute Percentage Error(MAPE)------------------------------------', MAPE)
        prophet_parameters = prophet_parameters.append({'MAPE': MAPE, 'MAAPE': MAAPE,
                                                        'seasonality_mode': p['seasonality_mode'],
                                                        'changepoint_prior_scale': p['changepoint_prior_scale'],
                                                        'holidays_prior_scale': p['holidays_prior_scale'],
                                                        'n_changepoints': p['n_changepoints'],
                                                        'fourier_order': p['fourier_order'],
                                                        'interval_width': p['interval_width']}, ignore_index=True)
        prophet_parameters.to_csv(csvpath + '/' + nome_arquivo_csv, sep='\t', encoding='utf-8')


def tuning_model(df_original
                 , mask1
                 , mask2
                 , threshold_date
                 , periodo=None
                 , frequencia=None
                 , cap=None
                 , floor=None
                 , interval_width=[0.8]
                 , loja=None
                 , tipo=None
                 , changepoint_prior_scale=[0.001, 0.01, 0.1, 0.5]
                 , seasonality_prior_scale=[0.01, 0.1, 1.0, 10]
                 , holidays_prior_scale=[0.01, 0.1, 1.0, 10]
                 , fourier_order=[10, 3, 4]):
    """ Realiza testes com os parâmetros especificados para o modelo, a fim de identificar combinação que obtém
        o melhor mape"""
    db = gendf.Database()
    nome_arquivo_csv = csvpath + '/model_parameters_' + loja + '_' + tipo + '.csv'
    params_grid = {'seasonality_mode': ('multiplicative', 'additive'),
                   'changepoint_prior_scale': changepoint_prior_scale,
                   'seasonality_prior_scale': seasonality_prior_scale,
                   'holidays_prior_scale': holidays_prior_scale
                   }
    # Generate all combinations of parameters
    all_params = [dict(zip(params_grid.keys(), val)) for val in itertools.product(*params_grid.values())]
    #grid = ParameterGrid(params_grid)
    df_treino = df_original.loc[mask1]
    df_test = df_original.loc[mask2]
    strt = df_test.ds.min()
    end = df_test.ds.max()

    prophet_parameters = pd.DataFrame(columns=['seasonality_mode'
        , 'seasonality_prior_scale'
        , 'changepoint_prior_scale'
        , 'holidays_prior_scale'
        , 'changepoint_range'
        , 'fourier_order'
        , 'interval_width'
        , 'MAPE'
        , 'MAE'
        , 'RMSE'
        , 'yearly'
        , 'yearly_prior'
        , 'weekly'
        , 'weekly_prior'
        , 'daily'
        , 'daily_prior'
        , 'hourly'
        , 'hourly_prior'])
    for p in all_params:
        test = pd.DataFrame()
        print(p)
        random.seed(0)
        train_model = Prophet(**params)
        #train_model.add_seasonality(name='yearly', period=365.25, fourier_order=fourier_order[0])
        train_model.add_seasonality(name='weekly', period=7, fourier_order=fourier_order[1])
        train_model.add_seasonality(name='daily', period=1, fourier_order=fourier_order[2])
        train_model.fit(df_treino)
        df_cv = cross_validation(train_model, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        #train_forecast = train_model.make_future_dataframe(periods=periodo, freq=frequencia)
        #train_forecast = train_model.predict(train_forecast)
        #test = train_forecast[['ds', 'yhat']]
        yearly = train_model.seasonalities.get('yearly')
        weekly = train_model.seasonalities.get('weekly')
        daily = train_model.seasonalities.get('daily')
        hourly = train_model.seasonalities.get('hourly')
        Actual = df_original[(df_original['ds'] > strt) & (df_original['ds'] <= end)]
        RMSE = round(fm.rmse(Actual['y'], test['yhat']), 3)
        MAPE = round(fm.mape(Actual['y'], test['yhat']) * 100, 3)
        MAE = round(fm.mae(Actual['y'], test['yhat']), 3)
        print('Root Mean Square Error(RMSE)------------------------------------', RMSE)
        print('Mean Absolute Percentage Error(MAPE)----------------------------', MAPE)
        print('Mean Absolute Error(MAE)----------------------------------------', MAE)
        prophet_parameters = prophet_parameters.append({'RMSE': RMSE
                                                           , 'MAPE': MAPE
                                                           , 'MAE': MAE
                                                           , 'seasonality_mode': p['seasonality_mode']
                                                           , 'seasonality_prior_scale': p['seasonality_prior_scale']
                                                           , 'changepoint_prior_scale': p['changepoint_prior_scale']
                                                           , 'holidays_prior_scale': p['holidays_prior_scale']
                                                           , 'changepoint_range': p['changepoint_range']
                                                           , 'interval_width': p['interval_width']
                                                           , 'prior_scale': p['prior_scale']
                                                           , 'yearly': yearly.get('fourier_order')
                                                           , 'yearly_prior': yearly.get('prior_scale')
                                                           , 'weekly': weekly.get('fourier_order')
                                                           , 'weekly_prior': weekly.get('prior_scale')
                                                           , 'daily': daily.get('fourier_order')
                                                           , 'daily_prior': daily.get('prior_scale')
                                                           , 'hourly': hourly.get('fourier_order')
                                                           , 'hourly_prior': hourly.get('prior_scale')}
                                                       , ignore_index=True)
        prophet_parameters.to_csv(nome_arquivo_csv, sep='\t', encoding='utf-8')
        db.insert_model_params(int(loja)
                               , str(train_model.growth)
                               , float(p['changepoint_range'])
                               , str(p['seasonality_mode'])
                               , float(yearly['fourier_order'])
                               , float(yearly['prior_scale'])
                               , float(weekly['fourier_order'])
                               , float(weekly['prior_scale'])
                               , float(daily['fourier_order'])
                               , float(daily['prior_scale'])
                               , float(hourly['fourier_order'])
                               , float(hourly['prior_scale'])
                               , float(p['seasonality_prior_scale'])
                               , int(train_model.mcmc_samples)
                               , float(p['interval_width'])
                               , int(train_model.uncertainty_samples)
                               , float(RMSE)
                               , float(MAE)
                               , float(MAPE))


def extract_date_features(df,timefield='ds'):
    """Extrai as features de data como colunas do dataframe.
          Parametros
          ----------
          df: Dataframe

          Retorna
          -------
          O dataframe acrescido das colunas mes, ano, diasemana, num (numero do dia), semana, diames hora e minuto
      """
    df = df.copy()
    try:
        if timefield == 'ds':
            df['mes'] = df[timefield].dt.strftime('%B')
            df['ano'] = df[timefield].dt.strftime('%Y')
            df['diasemana'] = df[timefield].dt.strftime('%A')
            df['num'] = df[timefield].dt.strftime('%w')
            df['semana'] = df[timefield].dt.strftime('%U')
            df['diames'] = df[timefield].dt.day
            df['hora'] = df[timefield].dt.hour
            df['minuto'] = df[timefield].dt.minute
            X = df[['mes', 'ano', 'diames', 'semana', 'diasemana', 'num', 'hora', 'minuto']]
            y = df['y']
            df_analise = pd.concat([X, y], axis=1)
        else:
            df['mes'] = df[timefield].dt.strftime('%B')
            df['ano'] = df[timefield].dt.strftime('%Y')
            df['diasemana'] = df[timefield].dt.strftime('%A')
            df['num'] = df[timefield].dt.strftime('%w')
            df['semana'] = df[timefield].dt.strftime('%U')
            df['diames'] = df[timefield].dt.day
            df_analise = df
    except AttributeError as dt_error:
        print(dt_error)
        return df

    return df_analise


def analyze_dataframe(df):
    """ Agrega os dados de um dataframe por ano, mes, semana e dia.

        Parametros
        ----------
        df: dataframe

        Retorno
        -------
        4 dataframes agregados por ano, mes, semana e dia respetivamente
    """
    df = extract_date_features(df)
    ano_agregado = pd.DataFrame(df.groupby("ano")["y"].sum()).reset_index().sort_values('y')
    mes_agregado = pd.DataFrame(df.groupby("mes")["y"].sum()).reset_index().sort_values('y')
    semana_agregado = pd.DataFrame(df.groupby("semana")["y"].sum()).reset_index().sort_values('y')
    dia_agregado = pd.DataFrame(df.groupby("diasemana")["y"].sum()).reset_index().sort_values('diasemana')
    return ano_agregado, mes_agregado, semana_agregado, dia_agregado


def configura_dataframe_treino_teste(df, inicio=None, fim=None, data_final=None):
    """Configura dos dataframes para treino e teste respetivamente retornando o dataframe de treino e o dataframe
    de teste"""
    # assert (data_final is None), 'Deve atribuir um valor para a data final (Atribua uma data válida' \
    #                               'para o parâmetro data_final)!'

    try:
        if inicio is None and fim is None:
            mascara_treino = (df['ds'] <= data_final)
            #print(mascara_treino.head(35))
            mascara_teste = (df['ds'] >= data_final)
        else:
            mascara_treino = (df['ds'] >= inicio) & (df['ds'] <= fim)
            mascara_teste = (df['ds'] >= fim) & (df['ds'] <= data_final)
    except ValueError as err:
        print("Deve existir a definição de inicio e fim que seja válida:\n" + err.args)

    df_treino = df.loc[mascara_treino]
    df_teste = df.loc[mascara_teste]
    return df_treino, df_teste


def split_dataframe(df, train_period=None):
    start = timeit.default_timer()
    # Remove negative values
    df = df[df['y'] > 0]
    try:
        df.reset_index(level=0, inplace=True)
    except ValueError as err:
        print('Columns on dataframe ', df.columns)
    finally:
        if df.index.name in df.columns:
            df.drop(columns=df.index.name, inplace=True)
            df.reset_index(level=0, inplace=True)

    # Split the data in proportion of 80% for training and 20% for test.
    if train_period is None:
        lastdayfrom = pd.to_datetime(df.at[df.shape[0]-1,'ds'])
        test_init_date = lastdayfrom - pd.Timedelta(days=7)
        mask = df['ds'] < test_init_date
        # Split the data and select `ds` and `y` columns.
        df_train = df[mask][['ds', 'y']]
        df_test = df[~ mask][['ds', 'y']]
        # Define threshold date.
        threshold_date=test_init_date
        #df_train = df.loc[:int(df.shape[0]*0.8)]
    # Split the data based on train_period parameter.
    else:
        lastdayfrom = pd.to_datetime(df.at[df.shape[0]-1,'ds'])
        test_init_date = lastdayfrom - pd.Timedelta(days=30)
        mask = df['ds'] < test_init_date
        # Split the data and select `ds` and `y` columns.
        df_train = df[mask][['ds', 'y']]
        df_test = df[~ mask][['ds', 'y']]
        # Define threshold date.
        threshold_date=test_init_date
    print('threshold_date: \n{}'.format(threshold_date))
    print("Completed in: {} seconds \n".format(timeit.default_timer() - start))
    return df_train, df_test, threshold_date


def split_df_month_week(df, start_time, end_time, train_period=None):
    """
        It splits the dataframe into train and test, based on params:

        df is the original dataframe
        train_period if it is not None. A period of 30 days (month)
        or 7 days (week). If train_period is None then split it in
        80% of data for training and 20% for test.
        When the train_period is set, it group the dataframe by month or week,
        splitting it in a list of dataframes grouped by train_period. The train
        and test dataframes are choosed by the frist two dataframes (0 and 1).
        threshold_date is defined by train max date in accordance with the
        train_period proportion (30 days, 7 days or 80% of data).
        It also remove zeroes from the dataframe.

        Returns the train and test dataframe and the defined threshold_date
        without zeroes.

    """
    start = timeit.default_timer()
    undesired_column = 'Unnamed: 0'
    print('Splitting for a {} days'.format(train_period))
    # Split the data in proportion of 80% for training and 20% for test.
    if train_period > 30:
        try:
            df_train = df.loc[:int(df.shape[0]*0.8)]
            # Define threshold date.
            threshold_date = pd.to_datetime(df_train['ds'].loc[:int(df_train.shape[0]*0.8)].max())
            df_test = df.loc[int(df.shape[0]*0.8+1):]
            print('threshold_date: \n{}'.format(threshold_date))
        except TypeError as typerr:
            print('reseting index ', typerr)
            df.reset_index(level=0, inplace=True)
            df_train = df.loc[:int(df.shape[0]*0.8)]
            threshold_date = pd.to_datetime(df_train['ds'].loc[:int(df_train.shape[0]*0.8)].max())
            df_test = df.loc[int(df.shape[0]*0.8+1):]
            print('threshold_date: \n{}'.format(threshold_date))
    else:
        try:
            df.reset_index(level=0, inplace=True)
        except ValueError as err:
            print('no need to reset index ', err)
        df.set_index('ds',inplace=True)
        # remove zero and negative values from the dataframe
        df = df[df['y'] > 0]
        if train_period == 30:
            DFList = [group[1] for group in df.groupby(df.index.month)]
            dfs_splitted = [0,1]
            for idx in dfs_splitted:
                try:
                    df_prep = dtclean.remove_zero_and_negative(DFList[idx], 'y')
                    df_prep['ds'] = pd.to_datetime(df_prep['ds'])
                    df_prep = df_prep.set_index(pd.DatetimeIndex(df_prep['ds']))
                    df_prep = df_prep.between_time(start_time, end_time)
                    if undesired_column in df_prep.columns:
                        df_prep.drop(columns=undesired_column, inplace=True)
                    if 'index' in df.columns:
                        df_prep.drop(columns='index', inplace=True)
                    if df_prep.index.name in df_prep.columns:
                        df_prep.drop(columns=df_prep.index.name, inplace=True)
                    if idx == 0:
                        df_train = df_prep
                    elif idx == 1:
                        df_test = df_prep
                except IndexError as ixerr:
                    print(ixerr)
                    df_train, df_test, threshold_date = split_dataframe(df)
        if train_period == 7:
            DFList = [group[1] for group in df.groupby(df.index.month)]
            try:
                df_prep = dtclean.remove_zero_and_negative(DFList[0], 'y')
                df_prep['ds'] = pd.to_datetime(df_prep['ds'])
                df_prep = df_prep.set_index(pd.DatetimeIndex(df_prep['ds']))
                df_prep = df_prep.between_time(start_time, end_time)
                if undesired_column in df_prep.columns:
                    df_prep.drop(columns=undesired_column, inplace=True)
                if 'index' in df.columns:
                    df_prep.drop(columns='index', inplace=True)
                if df_prep.index.name in df_prep.columns:
                    df_prep.drop(columns=df_prep.index.name, inplace=True)
            except IndexError as ixerr:
                print(ixerr)
                df_train, df_test, threshold_date = split_dataframe(df_prep)

            try:
                lastdayfrom = pd.to_datetime(df_prep.at[df_prep.shape[0]-1,'ds'])
                test_init_date = lastdayfrom - pd.Timedelta(days=7)
                mask = df_prep['ds'] < test_init_date
                # Split the data and select `ds` and `y` columns.
                df_train = df_prep[mask][['ds', 'y']]
                df_test = df_prep[~ mask][['ds', 'y']]
            except KeyError:
                df_prep.reset_index(level=0,inplace=True)
                lastdayfrom = pd.to_datetime(df_prep.at[df_prep.shape[0]-1,'ds'])
                test_init_date = lastdayfrom - pd.Timedelta(days=7)
                mask = df_prep['ds'] < test_init_date
                # Split the data and select `ds` and `y` columns.
                df_train = df_prep[mask][['ds', 'y']]
                df_test = df_prep[~ mask][['ds', 'y']]
    try:
        df_train.reset_index(level=0,inplace=True)
        df_test.reset_index(level=0,inplace=True)
    except ValueError as err:
            print('no need to reset index ', err)

    threshold_date = df_train['ds'].max()

    print('threshold_date: \n{}'.format(threshold_date))
    print("Completed in: {} seconds \n".format(timeit.default_timer() - start))
    return df_train, df_test, threshold_date


def detect_remove_outliers(df):
    df_anom = dout.detect_outliers(df, 'prophet')
    outliers = df_anom['anomaly'] == 1
    df_without_outliers = df_anom[~outliers][['ds', 'y']]
    dtview.plot_outliers(df, df_anom)
    return df_without_outliers


def remove_outliers(df, outliers):
    df.loc[df['y'].isin(outliers)]


def get_outliers(df, outliers):
    df_outliers = df.loc[df['ds', 'y'].isin(outliers)]
    return df_outliers


def normalize_dataframe(df):
    values = df.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized


def add_year_month_to_dataframe(df):
    df['year'] = [d.year for d in df.ds]
    df['month'] = [d.strftime('%b') for d in df.ds]
    years = df['year'].unique()
    return df


def analyze_ts_regularity():
    db = gendf.Database()
    df = make_ds.get_dataset(table='sisqual_all'
                             , columns=['store', 'ds', 'sales']
                             , col_to_rename='sales')
    size = [4000]
    rmax = [0.2, 0.45, 0.65]
    if not db.table_exists('entropy_analysis'):
        db.create_table_entropy_analysis()
    for store_number in df.store.unique():
        df_i = df.loc[df['store'] == store_number]
        store = store_number
        df_i.set_index('ds', inplace=True)
        df_i = df_i.resample('30min').sum()  # 15min -> 30min
        df_i.reset_index(inplace=True)
        entropy_table = db.dbase.Table('entropy_analysis', db.metadata, autoload=True, autoload_with=db.engine)
        statement = db.dbase.select([entropy_table.columns.store.distinct()]).where(
            entropy_table.columns.store == int(store))
        result_proxy = db.engine.execute(statement)
        result_set = result_proxy.fetchall()
        if len(result_set) <= 0:
            for n in size:
                if df_i.shape[0] > n:
                    df_i = df_i[:n]
                df_t_with_negative_values = df_i[df_i['y'] < 0]
                for rmx in rmax:
                    apen_spen = fm.evaluate_forecastability(df_i.y, m=2, r=rmx * np.std(df_i.y))
                    db.insert_forecastability(int(store)
                                              , df_i.shape[0]
                                              , int(df_t_with_negative_values['y'].count())
                                              , float(apen_spen['apen'])
                                              , float(apen_spen['spen']), 2, rmx)
                    dtview.print_df_entropy_test(df_i, apen_spen, rmx, df_t_with_negative_values, store=store)


def get_most_regular_store(nof_store=3, r_filter=0.2):
    """ Get the most regular store and the worst regular store according to Approximate Entropy calculated and
    stored on database
    -------------
    Parameters
    ------------
    nof_store = number of store to return as top and worst, default = 3, returns the top 3 stores with better ApEn
    and worst 3 with the higher ApEn on database
    r_filter = Allow to filter the r value calculated among 3 possibilities [0.2, 0.45, 0.65] default = 0.2
    """
    r_values = [0.2, 0.45, 0.65]
    db = gendf.Database()
    if r_filter not in r_values:
        r_filter = 0.2
        print("r_filter not in range {}.\n Using default (0.2)".format(r_values))
    entropy_table = db.dbase.Table('entropy_analysis', db.metadata, autoload=True, autoload_with=db.engine)
    statement = db.dbase.select([entropy_table.columns.store, db.dbase.func.min(entropy_table.columns.ApEn)
                                    , entropy_table.columns.r
                                    , entropy_table.columns.df_size]) \
        .group_by(entropy_table.columns.store, entropy_table.columns.r, entropy_table.columns.df_size) \
        .order_by(db.dbase.func.min(entropy_table.columns.ApEn))
    result_proxy = db.engine.execute(statement)
    result_set = result_proxy.fetchall()
    df = pd.DataFrame(result_set)
    df.columns = result_set[0].keys()
    df.rename(columns={'min_1': 'ApEn'}, inplace=True)
    df_top_tmp = df.loc[df['r'] == r_filter].groupby('ApEn').min().head(nof_store)
    df_worst_tmp = df.loc[df['r'] == r_filter].groupby('ApEn').min().tail(nof_store)
    top_reg_store = df_top_tmp.store.tolist()
    worst_reg_store = df_worst_tmp.store.tolist()
    return top_reg_store, worst_reg_store, df


def get_entropy_analysis(store=None):
    """ Get entropy analysis for store
    -------------
    Parameters
    ------------
    store = Store id to return
    """
    db = gendf.Database()
    if store is not None:
        entropy_table = db.dbase.Table('entropy_analysis', db.metadata, autoload=True, autoload_with=db.engine)
        statement = db.dbase.select([entropy_table.columns.store, entropy_table.columns.ApEn
                                        , entropy_table.columns.r
                                        , entropy_table.columns.df_size]) \
            .where(entropy_table.columns.store == store)
        result_proxy = db.engine.execute(statement)
        result_set = result_proxy.fetchall()
        df = pd.DataFrame(result_set)
        df.columns = result_set[0].keys()
        return df
    else:
        raise Exception


def get_forecast_analysis(store=None):
    """ Get forecast analysis store
    -------------
    Parameters
    ------------
    store = Store id to return
    """
    db = gendf.Database()
    if store is not None:
        forecast_analysis_table = db.dbase.Table('forecast_analysis', db.metadata, autoload=True, autoload_with=db.engine)
        statement = db.dbase.select([forecast_analysis_table.columns.analysis_moment
                , forecast_analysis_table.columns.store
                , forecast_analysis_table.columns.df_size
                , forecast_analysis_table.columns.ApEn
                , forecast_analysis_table.columns.fcst_period
                , forecast_analysis_table.columns.freq_hours
                , forecast_analysis_table.columns.work_hours
                , forecast_analysis_table.columns.mse_outlier
                , forecast_analysis_table.columns.rmse_outlier
                , forecast_analysis_table.columns.mae_outlier
                , forecast_analysis_table.columns.mape_outlier
                , forecast_analysis_table.columns.mse_without_outlier
                , forecast_analysis_table.columns.rmse_without_outlier
                , forecast_analysis_table.columns.mae_without_outlier
                , forecast_analysis_table.columns.mape_without_outlier
                , forecast_analysis_table.columns.model_outlier_file_name
                , forecast_analysis_table.columns.model_without_outlier_file_name]) \
            .where(forecast_analysis_table.columns.store == store)
        result_proxy = db.engine.execute(statement)
        result_set = result_proxy.fetchall()
        df = pd.DataFrame(result_set)
        df.columns = result_set[0].keys()
        return df
    else:
        forecast_analysis_table = db.dbase.Table('forecast_analysis', db.metadata, autoload=True, autoload_with=db.engine)
        statement = db.dbase.select([forecast_analysis_table.columns.analysis_moment
                , forecast_analysis_table.columns.store
                , forecast_analysis_table.columns.df_size
                , forecast_analysis_table.columns.ApEn
                , forecast_analysis_table.columns.fcst_period
                , forecast_analysis_table.columns.freq_hours
                , forecast_analysis_table.columns.work_hours
                , forecast_analysis_table.columns.mse_outlier
                , forecast_analysis_table.columns.rmse_outlier
                , forecast_analysis_table.columns.mae_outlier
                , forecast_analysis_table.columns.mape_outlier
                , forecast_analysis_table.columns.mse_without_outlier
                , forecast_analysis_table.columns.rmse_without_outlier
                , forecast_analysis_table.columns.mae_without_outlier
                , forecast_analysis_table.columns.mape_without_outlier
                , forecast_analysis_table.columns.model_outlier_file_name
                , forecast_analysis_table.columns.model_without_outlier_file_name])
        result_proxy = db.engine.execute(statement)
        result_set = result_proxy.fetchall()
        df = pd.DataFrame(result_set)
        df.columns = result_set[0].keys()
        return df



def save_model_performance(df, store, type, params):
    try:
        db = gendf.Database()
    except:
        pass
    for idx in range(len(df)):
        db.insert_model_performance(int(store)
                                    , str(type)
                                    , str(df['horizon'][idx])
                                    , float(df['mse'][idx])
                                    , float(df['rmse'][idx])
                                    , float(df['mae'][idx])
                                    , float(df['mape'][idx])
                                    , float(df['mdape'][idx])
                                    , float(df['coverage'][idx])
                                    , str(params))


def save_manual_evaluation(df, store, outlier):
    try:
        db = gendf.Database()
    except:
        pass
    for idx in range(len(df)):
        db.insert_manual_evaluation(int(store)
                                    , str(df['horizon'][idx])
                                    , float(df['rmse'][idx])
                                    , float(df['mae'][idx])
                                    , float(df['mape'][idx])
                                    , float(df['seasonality_prior_scale'][idx])
                                    , float(df['holidays_prior_scale'][idx])
                                    , float(df['changepoint_prior_scale'][idx])
                                    , int(outlier))



def make_forecast_positive(df, periods=90, freq='D', seas_mode='multiplicative'):
    # Fit the ProphetPos model
    try:
        if np.isinf(df['y'].values).any():
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["y"], how="all")
        model = pm.ProphetPos(growth='linear',
                              interval_width=0.8,
                              changepoint_prior_scale=0.5,
                              holidays_prior_scale=0.5,
                              seasonality_prior_scale=0.1,
                              changepoint_range=0.5,
                              seasonality_mode=seas_mode)
        model.add_seasonality(name='yearly', period=365.25, fourier_order=10, prior_scale=0.1)
        model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.5)
        model.add_seasonality(name='daily', period=1, fourier_order=3, prior_scale=0.5)
        model.add_seasonality(name='hourly', period=12, fourier_order=3, prior_scale=0.5)
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        future['ds'] = pd.to_datetime(future['ds'])
        future = future.set_index(pd.DatetimeIndex(future['ds']))
        future = future.between_time('10:00', '21:00')
        fcst = model.predict(future)
    except ValueError as err:
        print(err)
    return fcst, model, future


def apply_negative_binomial(df):
    # Negative binomial likelihood
    df['cap'] = 1.2 * df['y'].max()
    try:
        if np.isinf(df['y'].values).any():
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["y"], how="all")
        common_model_neg_binomial = Prophet(
            growth='logistic',
            seasonality_mode='multiplicative',
            likelihood='NegBinomial',
            changepoint_prior_scale=0.5,
        ).fit(df)
        future = common_model_neg_binomial.make_future_dataframe(90)
        future['cap'] = 1.2 * df['y'].max()
        fcst = common_model_neg_binomial.predict(future)
        return fcst, common_model_neg_binomial
    except TypeError as err:
        print(err)


def apply_log_transform(df):
    # Log-transform the data
    df['y'] = np.log(1 + df['y'])
    try:
        if np.isinf(df['y'].values).any():
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["y"], how="all")
        common_model_logtransform = Prophet(seasonality_mode='additive').fit(df)
        future = common_model_logtransform.make_future_dataframe(90)
        fcst = common_model_logtransform.predict(future)
        # Invert the transform
        common_model_logtransform.history['y'] = np.exp(common_model_logtransform.history['y']) - 1
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            fcst[col] = np.exp(fcst[col]) - 1
        return fcst, common_model_logtransform
    except ValueError as err:
        print(err)


def clip_trend(df):
    # Fit a usual prophet model, and clip
    common_model = Prophet(seasonality_mode='multiplicative').fit(df)
    future = common_model.make_future_dataframe(90)
    fcst = common_model.predict(future)
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        fcst[col] = fcst[col].clip(lower=0.0)
    return fcst, common_model


def apply_logistc_growth(df):
    # Fit a logistic growth model, and clip
    df['cap'] = 1.2 * df['y'].max()
    common_model_logistic = Prophet(
        growth='logistic',
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.5,
    ).fit(df)
    future = common_model_logistic.make_future_dataframe(90)
    future['cap'] = 1.2 * df['y'].max()
    fcst = common_model_logistic.predict(future)
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        fcst[col] = fcst[col].clip(lower=0.0)
    return fcst, common_model_logistic


def zeroes_negative_values(df, column_name):
    # zeroes the negative sales values
    df.loc[df[column_name] < 0, [column_name]] = 0
    return df
    # if df.isnull().sum() == 0:
    #     print('All negative values were zeroed successfully!')
    #     return df
    # else:
    #     print('Could not remove negative values. Are you sure there exists any?! Check it out!')


def resample_dataframe(df, res_freq):
    df.set_index('ds', inplace=True)
    df = df.resample(res_freq).sum()  # 15min -> 30min
    df = df.loc[df['y'] != 0]  # delete columns with 0 sales
    df.reset_index(inplace=True)
    return df


def get_cross_validation_metrics(store=40, type=0):
    """ Get metrics from database of defined store
    -------------
    Parameters
    ------------
    store = store identifier
    type = type of y variable, 0 (default) refers to client, 1 refers to sale
    """
    db = gendf.Database()
    cross_validation = db.dbase.Table('cross_validation', db.metadata, autoload=True, autoload_with=db.engine)
    statement = db.dbase.select([cross_validation.columns.analysis_moment
                                    , cross_validation.columns.type
                                    , cross_validation.columns.mae
                                    , cross_validation.columns.mse
                                    , cross_validation.columns.rmse
                                    , cross_validation.columns.mape
                                    , cross_validation.columns.mdape
                                    , cross_validation.columns.params]).where(cross_validation.columns.type == type)
    result_proxy = db.engine.execute(statement)
    result_set = result_proxy.fetchall()
    df = pd.DataFrame(result_set)
    df.columns = result_set[0].keys()
    return df


def parallelize_dataframe(df, func, num_partitions=None, num_cores=None):
    if num_partitions or num_cores is None:
        raise Exception
    else:
        df_split = np.array_split(df, num_partitions)
        pool = Pool(num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
    return df
