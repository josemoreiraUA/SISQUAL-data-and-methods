# *-- python --*
import logging
from src.data import data_explorer as dtexp, datacleaner as dtclean
from src import evaluate as fm
from itertools import product
from fbprophet import Prophet
import pandas as pd
import random
from sklearn.model_selection import ParameterGrid
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
# from fbprophet.plot import plot_forecast_component as plt_fcst
from fbprophet.plot import plot_cross_validation_metric, plot_seasonality
from tqdm import tqdm
logging.getLogger('fbprophet').setLevel(logging.ERROR)

def holidays(holidays_special:list, holidays_high: list):
    especial = pd.DataFrame({
        'holiday': 'especial',
        'ds': pd.to_datetime(holidays_special),
        'lower_window': 0,
        'upper_window': 0,
    })

    alta = pd.DataFrame({
        'holiday': 'alta',
        'ds': pd.to_datetime(holidays_high),
        'lower_window': 0,
        'upper_window': 0,
    })

    holidays = pd.concat((especial, alta))
    return holidays

def tune_prophet(df):
    param_grid = {  'growth': ["linear"],
                    #'changepoints': [0.8,0.9],
                    #'n_changepoints': [25, 50, 75],
                    'changepoint_range': [0.25, 0.5, 0.75, 0.9],
                    'yearly_seasonality': [False],
                    'weekly_seasonality': [True],
                    'daily_seasonality': [True],
                    #'holidays': [holidays],
                    'seasonality_mode': ["additive"],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                    #'holidays_prior_scale': [0.1, 1, 10],
                    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                    'mcmc_samples': [0],
                    'interval_width': [0.75, 0.95, 0.98],
                    'uncertainty_samples': [0]
                    }

    args = list(product(*param_grid.values()))
    args
    df_ps = pd.DataFrame()

    for arg in tqdm(args):
        m = Prophet(*arg[:7], arg[7](), *arg[8:]).fit(df)
        df_cv = cross_validation(m, initial='1000 days', period='30 days', horizon = '30 days')
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['params'] = str(arg)
        df_ps = df_ps.append(df_p)

    df_ps['mae+rmse'] = df_ps['mae']+df_ps['rmse']
    df_ps = df_ps.sort_values(['mae+rmse'])
    df_ps

def tuning_model(df, inicio=None, fim=None, periodo=None, frequencia=None,
                 interval_width=[0.98,0.95,0.90],
                 cap=None,
                 floor=None,
                 loja=None,
                 tipo=None,
                 changepoint_prior_scale=[15,20,25],
                 seasonality_prior_scale=[15,20,25],
                 holidays_prior_scale=[15,20,25],
                 n_changepoints=[30,50,66],
                 fourier_order=[10,20,30]):
    df_treino, df_teste = dtexp.configura_dataframe_treino_teste(df, inicio=inicio, fim=fim)
    df_original = df
    """ Realiza testes com os parâmetros especificados para o modelo, a fim de identificar combinação que obtém
        o melhor mape
        
        Parametros
        ----------
        
        df: Dataframe
        inicio: Data de início da TS
        fim: Data de fim da TS
        Periodo: periodo a ser utilizado no Prophet
        Frequencia: Frequencia de intervalo dos dados temporais
    """
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
        prophet_parameters.to_csv(nome_arquivo_csv, sep='\t', encoding='utf-8')
