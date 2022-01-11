import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.data import dataexplore as dtexp, datacleaner as dtclean, dataframegenerator as gendf
import src.forecast_metrics as fm
import settings as settings
from fbprophet import Prophet
import holidays
import seaborn as sns
import timeit

path, csvpath, imgpath, neg_values_path, large_file_path, model_path = settings.get_file_path()


def prophet_fit(df, prophet_model, today_index, lookback_days=None, predict_days=21
                , is_df_hourly=False
                ,init_hourly=None
                , finish_hourly=None
                , freq=None):
    """
    Fit the model to the time-series data and generate forecast for specified time frames

    Args
    ----

    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values

    prophet_model : Prophet model
        Prophet model with configured parameters

    today_index : int
        The index of the date list in the df dataframe, where Day (today_index-lookback_days)th
        to Day (today_index-1)th is the time frame for training

    lookback_days: int, optional (default=None)
        As described above, use all the available dates until today_index as
        training set if no value assigned

    predict_days: int, optional (default=21)
        Make prediction for Day (today_index)th to Day (today_index+predict_days)th

    Returns
    -------
    fig : matplotlib Figure
        A plot with actual data, predicted values and the interval
    forecast : pandas DataFrame
        The predicted result in a format of dataframe
    prophet_model : Prophet model
        Trained model
    """

    # segment the time frames
    baseline_ts = df['ds'][:today_index]
    baseline_y = df['y'][:today_index]
    if not lookback_days:
        print('Training period from {} to {} ({} days)'.format(df['ds'][0],
                                                                     df['ds'][today_index - 1],
                                                                     today_index))
    else:
        baseline_ts = df['ds'][today_index - lookback_days:today_index]
        baseline_y = df.y[today_index - lookback_days:today_index]
        print('Training period from {} to {} ({} days)'.format(df['ds'][today_index - lookback_days],
                                                                     df['ds'][today_index - 1],
                                                                     lookback_days))
    print('Forecasting from {} to {} ({} days)'.format(df['ds'][today_index],
                                                   df['ds'][today_index + predict_days - 1],
                                                   predict_days))

    # fit the model
    prophet_model.fit(pd.DataFrame({'ds': baseline_ts.values,
                                    'y': baseline_y.values}))
    future = prophet_model.make_future_dataframe(periods=predict_days, freq=freq)
    # make prediction
    if is_df_hourly:
        future['ds'] = pd.to_datetime(future['ds'])
        future = future.set_index(pd.DatetimeIndex(future['ds']))
        future = future.between_time(init_hourly, finish_hourly)
    forecast = prophet_model.predict(future)
    # generate the plot
    fig = prophet_model.plot(forecast)
    return fig, forecast, prophet_model


def prophet_plot(df, fig, today_index, lookback_days=None, predict_days=21, outliers=list()):
    """
    Plot the actual, predictions, and anomalous values

    Args
    ----

    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values

    fig : matplotlib Figure
        A plot with actual data, predicted values and the interval which we previously obtained
        from Prophet's model.plot(forecast).

    today_index : int
        The index of the date list in the dataframe dividing the baseline and prediction time frames.

    lookback_days : int, optional (default=None)
        Day (today_index-lookback_days)th to Day (today_index-1)th is the baseline time frame for training.

    predict_days : int, optional (default=21)
        Make prediction for Day (today_index)th to Day (today_index+predict_days)th.

    outliers : a list of (datetime, int) tuple
        The outliers we want to highlight on the plot.
    """
    # retrieve the subplot in the generated Prophets matplotlib figure
    ax = fig.get_axes()[0]

    start = 0
    end = today_index + predict_days
    x_pydatetime = df['ds'].dt.to_pydatetime()
    # highlight the actual values of the entire time frame
    ax.plot(x_pydatetime[start:end],
            df.y[start:end],
            color='orange', label='Actual')

    # plot each outlier in red dot and annotate the date
    for outlier in outliers:
        ax.scatter(outlier[0], outlier[1], color='red', label='Outlier')
        ax.text(outlier[0], outlier[1], str(outlier[0])[:10], color='red')

    # highlight baseline time frame with gray background
    if lookback_days:
        start = today_index - lookback_days
    ax.axvspan(x_pydatetime[start],
               x_pydatetime[today_index],
               color=sns.xkcd_rgb['grey'],
               alpha=0.2)

    # annotate the areas, and position the text at the bottom 5% by using ymin + (ymax - ymin) / 20
    ymin, ymax = ax.get_ylim()[0], ax.get_ylim()[1]
    ax.text(x_pydatetime[int((start + today_index) / 2)], ymin + (ymax - ymin) / 20, 'Baseline Area')
    ax.text(x_pydatetime[int((today_index * 2 + predict_days) / 2)], ymin + (ymax - ymin) / 20, 'Forecasting Area')

    # re-organize the legend
    patch1 = mpatches.Patch(color='red', label='Outlier')
    patch2 = mpatches.Patch(color='orange', label='Actual')
    patch3 = mpatches.Patch(color='skyblue', label='Forecasting and interval')
    patch4 = mpatches.Patch(color='grey', label='Baseline Area')
    plt.legend(handles=[patch1, patch2, patch3, patch4])
    plt.show()


def get_outliers(df, forecast, today_index, predict_days=21):
    """
    Combine the actual values and forecast in a data frame and identify the outliers

    Args
    ----

    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values

    forecast : pandas DataFrame
        The predicted result in a dataframe which was previously generated by
        Prophet's model.predict(future)

    today_index : int
        The summary statistics of the right tree node.

    predict_days : int, optional (default=21)
        The time frame we segment as prediction period

    Returns
    -------
    outliers : a list of (datetime, int) tuple
        A list of outliers, the date and the value for each
    df_pred : pandas DataFrame
        The data set contains actual and predictions for the forecast time frame
    """
    df_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(predict_days)
    df_pred.index = df_pred['ds'].dt.to_pydatetime()
    df_pred.columns = ['ds', 'preds', 'lower_y', 'upper_y']
    df_pred['actual'] = df['y'][today_index: today_index + predict_days].values

    # construct a list of outliers
    outlier_index = list()
    outliers = list()
    for i in range(df_pred.shape[0]):
        actual_value = df_pred['actual'][i]
        if actual_value < df_pred['lower_y'][i] or actual_value > df_pred['upper_y'][i]:
            outlier_index += [i]
            outliers.append((df_pred.index[i], actual_value))
            # optional, print out the evaluation for each outlier
            print('=====')
            print('Actual value {} out of predicted interval'.format(actual_value))
            print('Interval: de {} até {}'.format(df_pred['lower_y'][i], df_pred['upper_y'][i]))
            print('Date: {}'.format(str(df_pred.index[i])[:10]))

    return outliers, df_pred


def set_most_freq_hours(df, freq_hours, not_default_hours=False):
    start_time = '10:00'
    end_time = '19:00'
    df_hours = pd.DataFrame(df[df['y'] > 0].groupby(df['ds'].dt.hour)['ds'].count())
    most_common_hours = df_hours[df_hours['ds'] >= freq_hours]
    list_hours = most_common_hours.index.values
    if len(list_hours) > 1 and not_default_hours:
        start_time = str(list_hours[0])+':00'
        end_time = str(list_hours[len(list_hours)-1])+':00'

    return start_time, end_time

def prophet_predict_grp(df,periods_factor=1, hyperparams=False
, train_period=None):
    start = timeit.default_timer()
    undesired_column = 'Unnamed: 0'
    start_time = '10:00'
    end_time = '19:00'
    forecast_period={'week': 7, 'month': 30, 'records': df.shape[0]}

    if train_period == 'm':
        fcst_period=forecast_period['month']
    elif train_period == 'w':
        fcst_period=forecast_period['week']
    elif train_period == 'r':
        fcst_period=forecast_period['records']
    elif train_period is None:
        print('Train period cannot be None')

    df_trn, df_tst, threshold_date = dtexp.split_df_month_week(df, start_time
    , end_time, train_period=fcst_period)

    alpha=0.8
    if hyperparams:
        mdl = Prophet(growth = 'linear',
                      interval_width=alpha,
                      changepoint_prior_scale=0.5,
                      daily_seasonality=False,
                      #holidays_prior_scale=0.1,
                      seasonality_prior_scale=10,
                      #changepoint_range=0.5,
                      seasonality_mode="additive")
        mdl.add_country_holidays(country_name='ESP')
        mdl.add_seasonality(name='weekly', period=7, fourier_order=5)#, prior_scale=0.5)
        mdl.add_seasonality(name='daily', period=1, fourier_order=5)#, prior_scale=0.5)
    else:
        mdl = Prophet()
        mdl.add_country_holidays(country_name='ESP')

    mdl.fit(df_trn)
    future = mdl.make_future_dataframe(periods=fcst_period*periods_factor, freq='30min')

    future['ds'] = pd.to_datetime(future['ds'])
    future = future.set_index(pd.DatetimeIndex(future['ds']))
    forecast = mdl.predict(future)
    print("Completed in: {} seconds \n".format(timeit.default_timer() - start))


    return df_trn, df_tst, forecast, mdl, threshold_date, fcst_period


def prophet_predict(df,periods_factor=1, hyperparams=False, train_period=None):
    start = timeit.default_timer()
    undesired_column = 'Unnamed: 0'
    start_time = '09:00'
    end_time = '21:00'
    spain_holidays = holidays.ES
    forecast_period={'week': 7, 'month': 30, 'records': df.shape[0]}

    if train_period == 'm':
        fcst_period=forecast_period['month']
    else:
        fcst_period=7
    #elif train_period == 'w':
    #    fcst_period=forecast_period['week']
    #elif train_period == 'r':
    #    fcst_period=forecast_period['records']
    #elif train_period is None:
    #    print('Train period cannot be None')

    #if df_prep.shape[0] < 6000:
    df_trn, df_tst, threshold_date = dtexp.split_dataframe(df, train_period=fcst_period)
    #else:
    #    periods = df_prep.ds.max() - df_prep.ds.min()
    #    if periods.days < 60:
    #        fcst_period=forecast_period['week']
    #        df_trn, df_tst, threshold_date = dtexp.split_dataframe(df_prep, train_period=forecast_period['week'])
    #    else:
    #        df_trn, df_tst, threshold_date = dtexp.split_dataframe(df_prep, train_period=forecast_period['month'])

    alpha=0.8
    if hyperparams:
        mdl = Prophet(growth = 'linear',
                      interval_width=alpha,
                      changepoint_prior_scale=0.1,
                      holidays_prior_scale=0.1,
                      seasonality_prior_scale=0.001,
                      #changepoint_range=0.5,
                      seasonality_mode="additive")
        mdl.add_seasonality(name='weekly', period=7, fourier_order=3)
        mdl.add_seasonality(name='daily', period=1, fourier_order=4)
    else:
        mdl = Prophet(holidays=holidays.ES)
    mdl.fit(df_trn)
    future = mdl.make_future_dataframe(periods=fcst_period*periods_factor, freq='30min')

    future['ds'] = pd.to_datetime(future['ds'])
    future = future.set_index(pd.DatetimeIndex(future['ds']))
    forecast = mdl.predict(future)
    print("Completed prediction in: {} seconds \n".format(timeit.default_timer() - start))


    return df_trn, df_tst, forecast, mdl, threshold_date, fcst_period


def params_analysis(df, max_params=False, add_regressor=False
                    , fcst_year=False, year='2017'
                    , period=180, frequency='D'
                    , prior=10):
    start = timeit.default_timer()
    undesired_column = 'Unnamed: 0'
    start_time = '09:00'
    end_time = '21:00'
    prior
    spain_holidays = holidays.ES

    df['year'] = [d.year for d in df.ds]
    if fcst_year:
        df = df.query('year <= '+year)
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    df_trn, df_tst, threshold_date = dtexp.split_dataframe(df, train_period=None)

    alpha=0.8
    if max_params:
        mdl = Prophet(growth = 'linear',
                      interval_width=alpha,
                      #changepoint_prior_scale=prior,
                      holidays_prior_scale=prior,
                      #seasonality_prior_scale=prior,
                      #yearly_seasonality=False,
                      seasonality_mode="additive")
        #mdl.add_seasonality(name='yearly', period=365.25, fourier_order=10)
        #mdl.add_seasonality(name='weekly', period=7, fourier_order=3)
        #mdl.add_seasonality(name='daily', period=1, fourier_order=4)
        mdl.add_country_holidays(country_name='ESP')

    elif add_regressor:
        mdl = Prophet(growth = 'linear',
                      interval_width=alpha,
                      #holidays=holidays.ES,
                      #changepoint_prior_scale=changepoint_prior_scale_min,
                      #holidays_prior_scale=prior_scale_min,
                      seasonality_prior_scale=prior_scale_min,
                      seasonality_mode="additive")
        for i, month in enumerate(months):
            df_trn[month] = (df_trn['ds'].dt.month == i + 1).values.astype('float')
            df_tst[month] = (df_tst['ds'].dt.month == i + 1).values.astype('float')
        mdl.add_regressor(month)
    else:
        mdl = Prophet()

    mdl.fit(df_trn)
    future = mdl.make_future_dataframe(periods=period, freq=frequency)


    future['ds'] = pd.to_datetime(future['ds'])
    future = future.set_index(pd.DatetimeIndex(future['ds']))
    if add_regressor:
        for i, month in enumerate(months):
            future[month] = (future['ds'].dt.month == i + 1).values.astype('float')
    forecast = mdl.predict(future)
    print("Completed prediction in: {} seconds \n".format(timeit.default_timer() - start))


    return forecast, mdl, df_trn, df_tst, threshold_date


def prophet_manual_evaluate_parameters(store, df, horizon, outlier, periods_factor=1
, train_period=None):
    nome_arquivo_csv=store+'_manual_parameters_evaluation.csv'
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    mapes = []  # Store the MAPEs for each params here
    maes = []  # Store the MAEs for each params here

    undesired_column = 'Unnamed: 0'
    start_time = '10:00'
    end_time = '19:00'
    forecast_period={'week': 7, 'month': 30, 'records': df.shape[0]}

    if train_period == 'm':
        fcst_period=forecast_period['month']
    elif train_period == 'w':
        fcst_period=forecast_period['week']
    elif train_period == 'r':
        fcst_period=forecast_period['records']
    elif train_period is None:
        print('Train period cannot be None')

    df_trn, df_tst, threshold_date = dtexp.split_df_month_week(df, start_time
    , end_time, train_period=fcst_period)

    # Manually evaluate all parameters
    for params in all_params:
        mdl = Prophet(**params)  # Create model with given params
        mdl.add_seasonality(name='weekly', period=7, fourier_order=3)
        mdl.add_seasonality(name='daily', period=1, fourier_order=4)
        mdl.fit(df_trn)
        future = mdl.make_future_dataframe(periods=fcst_period*periods_factor, freq='30min')
        future['ds'] = pd.to_datetime(future['ds'])
        future = future.set_index(pd.DatetimeIndex(future['ds']))
        forecast = mdl.predict(future)
        df_tst_s = df_tst.copy()
        df_tst_s['ds'] = pd.to_datetime(df_tst_s['ds'])
        df_tst_s = df_tst_s.set_index(pd.DatetimeIndex(df_tst_s['ds']))
        df_tst_forecast = mdl.predict(df_tst_s)
        df_tst_forecast = dtclean.remove_zero_and_negative(df_tst_forecast, 'yhat')
        if df_tst_forecast.shape[0] < df_tst_s.shape[0]:
            mask = df_tst_s['ds'].isin(df_tst_forecast['ds'])
            df_tst_s = df_tst_s[mask][['ds', 'y']]
        mape = fm.mape(np.array(df_tst_s['y']),np.array(df_tst_forecast['yhat']))
        mae = fm.mae(np.array(df_tst_s['y']),np.array(df_tst_forecast['yhat']))
        rmse = fm.rmse(np.array(df_tst_s['y']),np.array(df_tst_forecast['yhat']))
        rmses.append(round(rmse,3))
        mapes.append(round(mape*100,3))
        maes.append(round(mae,3))

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results['mape'] = mapes
    tuning_results['mae'] = maes
    tuning_results['horizon'] = horizon
    db = gendf.Database()
    dtexp.save_manual_evaluation(tuning_results, store, outlier)
    tuning_results.to_csv(csvpath + '/' + nome_arquivo_csv, sep='\t', encoding='utf-8')

    return tuning_results
