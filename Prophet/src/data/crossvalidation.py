# Python
import itertools
import numpy as np
import pandas as pd
import settings as settings
import src.forecast_metrics as fm
from src.data import dataexplore as dtexp
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics


def cross_validate(model
                   , store
                   , type
                   , initial='730 days'
                   , period='180 days'
                   , horizon='365 days'):
    prophet_parameters = {'growth': model.growth
        , 'interval_width': model.interval_width
        , 'seasonality_mode': model.seasonality_mode
        , 'seasonality_prior_scale': model.seasonality_prior_scale
        , 'changepoint_prior_scale': model.changepoint_prior_scale
        , 'holidays_prior_scale': model.holidays_prior_scale
        , 'changepoint_range': model.changepoint_range}
    params = str(prophet_parameters)
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)
    try:
        df_p['mape']
    except KeyError as kerr:
        df_p['mape'] = 0
    dtexp.save_model_performance(df_p, store, type, params)


def cross_validate_parallel(store, df, horizons, mdlcutoffs):
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here
        mapes = []  # Store the MAPEs for each params here
        maes = []  # Store the MAEs for each params here
        # Use cross validation to evaluate all parameters
        for params in all_params:
            model = Prophet(**params).fit(df)  # Fit model with given params
            print(type(mdlcutoffs))
            #print(mdlcutoffs)
            df_cv = cross_validation(model, cutoffs=mdlcutoffs.tolist(), horizon=horizons, parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            print('RMSE: \n'.format(df_p['rmse']))
            print('MAPE: \n'.format(df_p['mape']))
            print('MAE: \n'.format(df_p['mae']))
            rmses.append(df_p['rmse'].values[0])
            mapes.append(df_p['mape'].values[0])
            maes.append(df_p['mae'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        tuning_results['mape'] = mapes
        tuning_results['mae'] = maes
        print(tuning_results)
        try:
            df_p['mape']
        except KeyError as kerr:
            df_p['mape'] = 0
        model_params = get_prophet_model_params(model)
        #dtexp.save_model_performance(df_p, store, type, model_params)


def get_prophet_model_params(model):
    prophet_parameters = {'growth': model.growth
        , 'interval_width': model.interval_width
        , 'seasonality_mode': model.seasonality_mode
        , 'seasonality_prior_scale': model.seasonality_prior_scale
        , 'changepoint_prior_scale': model.changepoint_prior_scale
        , 'holidays_prior_scale': model.holidays_prior_scale
        , 'changepoint_range': model.changepoint_range}
    model_params = str(prophet_parameters)
    return model_params
