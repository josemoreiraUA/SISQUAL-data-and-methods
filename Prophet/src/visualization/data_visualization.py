import matplotlib.colors
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from fbprophet.plot import add_changepoints_to_plot, plot_seasonality as plt_seas
from fbprophet import Prophet
from beautifultable import BeautifulTable
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
import settings as settings
from src.data import utils as utls
from src.data import dataexplore as dtexp
from src.data import dataframegenerator as gendf, make_dataset as make_ds
import pandas as pd
import seaborn as sns
import calplot
import arviz as az
import numpy as np
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython import display
#from matplotlib import rcParams
#rcParams['animation.convert_path'] = r'/usr/local/Cellar/imagemagick/convert'
#rcParams['animation.ffmpeg_path'] = r'/usr/local/Cellar/ffmpeg\bin/ffmpeg.exe'

sns_c = sns.color_palette(palette='deep')
path, csvpath, imgpath, neg_values_path, large_file_path, model_path = settings.get_file_path()


def compare_dataframes_with_bar(df1, df2, nametrace1=None, nametrace2=None, idx_start=0, idx_end=28):
    trace1 = go.Bar(x=df1['ds'].iloc[idx_start:idx_end], y=df1['y'], name=nametrace1)
    trace2 = go.Bar(x=df2['ds'].iloc[idx_start:idx_end], y=df2['yhat'], name=nametrace2)
    data = [trace1, trace2]
    layout = go.Layout(barmode='group')
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )
    return fig


def compare_dataframes_with_scatter(df_1, df_2, title=None, mode=None,
                                    name1=None,
                                    name2=None,
                                    mape=None,
                                    maape=None,
                                    annotation_data_point=None,
                                    is_forecast=False):
    if mode is None:
        mode = 'lines'
    text = ["Text G", "Text H", "Text I"],
    textposition = "bottom center"
    trace_treino = go.Scatter(x=df_1['ds'], y=df_1['y'], mode=mode, name=name1)
    if is_forecast:
        trace_teste = go.Scatter(x=df_2.ds, y=df_2.yhat, mode=mode, name=name2)
    else:
        trace_teste = go.Scatter(x=df_2['ds'], y=df_2['y'], mode=mode, name=name2)
    data = [trace_treino, trace_teste]
    layout = go.Layout(height=500,
                       width=1000,
                       title={'text': title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data, layout=layout)
    if mape is not None and maape is not None:
        fig.add_annotation(x=annotation_data_point, y=1,
                           text="Mape: " + mape + " Maape: " + maape,
                           showarrow=False,
                           arrowhead=1)
    if plt is not None:
        utls.save_figure(imgpath, "df_with_scatter", plt)
    else:
        utls.save_figure(imgpath, "df_with_scatter", fig, isfigure=True)

    return fig


def plot_forecast(forecast, model, filename=None, df_test=None, title=None):
    if filename is None:
        filename = "forecast_"
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    if df_test is not None:
        ax.scatter(df_test.ds, df_test['y'], color='r')
    fig = model.plot(forecast, ax)
    if title is not None:
        ax.set_title(title)
    utls.save_figure(imgpath, filename, plt)
    return plt, fig


def plot_forecast_vs_actuals(fcst, df, filename=None, titlecomplement=None):
    title = 'Forecast vs Actuals'
    if titlecomplement is not None:
        title = title + titlecomplement
    if filename is None:
        filename = "forecast_vs_actuals_"
    f, ax = plt.subplots(figsize=(14, 5))
    f.set_figheight(5)
    f.set_figwidth(15)
    df.plot(kind='line', x='ds', y='y', color='red', label='Test', ax=ax)
    fcst.plot(kind='line', x='ds', y='yhat', label='Forecast', ax=ax)
    plt.title(title)
    utls.save_figure(imgpath, filename, plt)
    return plt, f


def plot_dataframe_as_table(df):
    trace = go.Table(header=dict(values=list(df.columns)),
                     cells=dict(values=[df.ds, df.y]))

    data = [trace]
    fig = go.Figure(data=data)
    return fig


def plot_forecast_as_table(df):
    trace_df = go.Table(header=dict(values=list(df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])),
                        cells=dict(values=[df.ds,
                                           df.yhat,
                                           df.yhat_lower,
                                           df.yhat_upper]))
    data = [trace_df]
    fig_df = go.Figure(data=data)
    return fig_df


def plot_scatter(df, title=None):
    data = go.Scatter(x=df.ds, y=df.y)
    layout = go.Layout(height=500,
                       width=1000,
                       title={'text': title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=[data], layout=layout)
    utls.save_figure(imgpath, "scatter", fig, isfigure=True)
    return fig


def plot_df_features(df, fig, ax, title='Análise'):
    palette = sns.color_palette("mako_r", 4)
    a = sns.barplot(x="mes", y="y", hue='mes', data=df)
    a.set_title("Dados de " + title, fontsize=15)
    plt.legend(loc='upper right')
    plt.show()
    utls.save_figure(imgpath, "df_features", plt)
    return plt


def plot_df_features_as_table(df):
    trace = go.Table(header=dict(values=list(df.columns)),
                     cells=dict(values=[df.mes,
                                        df.diames,
                                        df.semana,
                                        df.diasemana,
                                        df.num,
                                        df.hora,
                                        df.minuto,
                                        df.y]))
    data = [trace]
    fig = go.Figure(data=data)
    return fig

def plot_outliers(df, df_anom, filename=None):
    if filename is None:
        filename = "outliers_detected_"
    # visualize the results
    fig, ax = plt.subplots(figsize=(10, 6))
    a = df.loc[df_anom['anomaly'] == 1, ['ds', 'y']]  # anomaly
    ax.plot(df_anom['ds'], df_anom['y'], color='blue')
    ax.scatter(a['ds'], a['y'], color='red')
    utls.save_figure(imgpath, filename, plt)

def check_outliers_univariate(df, variable="Quantity"):
    '''Cria uma figura boxplot com os dados do dataframe e retorna uma figura'''
    trace1 = go.Box(y=df['y'],
                    boxpoints='outliers',
                    marker_color='rgb(107,174,214)',
                    line_color='rgb(107,174,214)',
                    name=variable,
                    boxmean='sd')
    data = [trace1]
    fig = go.Figure(data)
    if plt is not None:
        utls.save_figure(imgpath, "outliers_univariate", plt)
    else:
        utls.save_figure(imgpath, "outliers_univariate", fig, isfigure=True)
    return fig


def check_outliers_multivariate(df):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(df['ds'], df['y'])
    ax.set_xlabel('Data e hora do registo')
    ax.set_ylabel('Quantidade')
    if plt is not None:
        utls.save_figure(imgpath, "outliers_multivariate", plt)
    else:
        utls.save_figure(imgpath, "outliers_multivariate", fig, isfigure=True)
    return fig


def view_optimum_parameters(nome_arquivo):
    parameters_df = pd.read_csv(nome_arquivo, sep='\t')
    parameters_df = parameters_df.sort_values(by=['MAPE'])
    parameters_df = parameters_df.reset_index(drop=True)
    parameters_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    trace_df_parameters = go.Table(header=dict(values=list(parameters_df[['MAPE', 'Parameters']])),
                                   cells=dict(values=[parameters_df.MAPE,
                                                      parameters_df.Parameters]))
    data = [trace_df_parameters]
    fig_df_parameters = go.Figure(data=data)
    if plt is not None:
        utls.save_figure(imgpath, "optimum_parameters", plt)
    else:
        utls.save_figure(imgpath, '"optimum_parameters"', fig_df_parameters, isfigure=True)
    return fig_df_parameters


def plot_dados_agregados_semana_mes(
        df,
        tipo_agregacao=None,
        title=None,
        xaxes_title=None,
        yaxis_title=None,
        hover=None,
        is_client_df=False):
    if is_client_df:
        yhover = 449
    else:
        yhover = 5704
    if tipo_agregacao == 's':
        agregacao = 'semana'
    elif tipo_agregacao == 'm':
        agregacao = 'mes'
    elif tipo_agregacao == 'a':
        agregacao = 'ano'
    elif tipo_agregacao == 'd':
        agregacao = 'diasemana'
    trace = go.Bar(x=df[agregacao],
                   y=df.y,
                   name=title,
                   marker_color=df.y)
    data = [trace]
    layout = go.Layout(barmode='group',
                       title={'text': title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(title_text=xaxes_title, showgrid=True, ticks="outside", tickson="boundaries", ticklen=20)
    fig.update_yaxes(title_text=yaxis_title)

    if hover is not None:
        fig.update_layout(
            showlegend=False,
            annotations=[
                dict(
                    x=22,
                    y=yhover,
                    xref="x",
                    yref="y",
                    text=hover,
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                    ),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=20,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                )
            ]
        )
    # if plt is not None:
    #     utls.save_figure(imgpath, "dados_agregados_semana_mes", plt)
    # else:
    utls.save_figure(imgpath, "dados_agregados_semana_mes", fig, isfigure=True)
    return fig


def plot_results(x_values, y_values, title=None, xaxes_title=None, yaxis_title=None):
    trace_mape = go.Bar(x=x_values, y=y_values, name=title, marker_color=y_values)
    data_mape = [trace_mape]
    layout = go.Layout(barmode='group',
                       title={'text': title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig_mape = go.Figure(data=data_mape, layout=layout)
    fig_mape.update_xaxes(title_text=xaxes_title, showgrid=True, ticks="outside", tickson="boundaries", ticklen=20)
    fig_mape.update_yaxes(title_text=yaxis_title)
    # if plt is not None:
    #     utls.save_figure(imgpath, "mape_results", plt)
    # else:
    utls.save_figure(imgpath, "mape_results", fig_mape, isfigure=True)
    return fig_mape


def plot_total_dataframe_data(df1,
                              df2,
                              title=None,
                              title2=None,
                              labels_1=None,
                              labels_2=None,
                              annot_text_1=None,
                              annot_text_2=None):
    colors = ['orange', 'mediumturquoise']
    fig = go.Figure(data=[go.Pie(labels=labels_1,
                                 values=[df1['y'].count(), df2['y'].count()])])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(
        title_text=title,
        annotations=[dict(text=annot_text_1, x=0.12, y=1, font_size=20, showarrow=False),
                     dict(text=annot_text_2, x=0.87, y=1, font_size=20, showarrow=False)])

    fig.show()

    colors = ['lightturquoise', 'red']

    fig = go.Figure(data=[go.Pie(labels=labels_2,
                                 values=[np.sum((df1['y'] < 0).values.ravel()),
                                         np.sum((df2['y'] < 0).values.ravel())])])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(
        title_text=title2)

    utls.save_figure(imgpath, "total_df_data", fig, isfigure=True)
    return fig


def plot_timeseries(df_src, title, color):
    df = df_src.groupby('ds').sum()
    df.reset_index(level=0, inplace=True)
    df.ds = pd.to_datetime(df.ds)
    df.index = df.ds
    df.y.plot(figsize=(12, 8), color=color)
    plt.title(title)
    utls.save_figure(imgpath, "timeseries", plt)
    return plt


def print_df_entropy_test(df, results, r_value, df_neg=None, store=0):
    table = BeautifulTable()
    if df_neg is not None:
        table.rows.append([store
                              , str(results['apen'])
                              , str(results['spen'])
                              , str(df_neg['y'].count())
                              , str(df.shape[0])
                              , str(r_value)])
        table.columns.header = ["Store", "ApEn", "SpEn", "Negative values", "Dataframe size", 'r value']
    else:
        table.rows.append([store
                              , str(results['apen'])
                              , str(results['spen'])
                              , str(df.shape[0])
                              , str(r_value)])
        table.columns.header = ["store", "ApEn", "SpEn", "Dataframe size", 'r value']
    print(table)


def print_dataframe_core_info(df):
    table = BeautifulTable()
    table.rows.append([str(df['ds'].min()), str(df['ds'].max()), df['ds'].max() - df['ds'].min()])
    table.columns.header = ["Sales Dataframe Start", "Sales Dataframe End", "Data Period Range"]
    print(table)


def plot_bp_trend_seasonality(df):
    """ plot da tendência e sazonalidade em visualização tipo Box Plot

        Parametros
        ----------

        df: dataframe
    """
    df = dtexp.add_year_month_to_dataframe(df)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
    sns.boxplot(x='year', y='y', data=df, ax=axes[0])
    sns.boxplot(x='month', y='y', data=df.loc[~df.year.isin([2015, 2020]), :])
    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18);
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    utls.save_figure(imgpath, "trend_boxplot", plt)
    plt.show()


def plot_added_changepoints(model=None, fcst=None):
    """ Plot os changepoints adicionados

        Parametros
        ----------
        model: Modelo do Prophet
        fcst: dataframe previsto
    """
    if model is None or fcst is None:
        raise Exception('Must inform Prophet model and forecast dataframe')
    else:
        fig = model.plot(fcst)
        a = add_changepoints_to_plot(fig.gca(), model, fcst)
        utls.save_figure(imgpath, "ts_with_added_changepoints", plt)


def plot_prophet_model_changepoints(model=None, fcst=None):
    """ Plot os changepoints do modelo do Prophet (default = 25 changepoints)

        Parametros
        ----------
        model: Modelo do Prophet
        fcst: dataframe previsto
    """
    if model is None or fcst is None:
        raise Exception('Must inform Prophet model and forecast dataframe')
    else:
        fig = model.plot(fcst)
        for changepoint in model.changepoints:
            plt.axvline(changepoint, ls='--', lw=1)
        utls.save_figure(imgpath, 'complete_time_series_with_changepoints_', plt)


def plot_train_test_split(df_train, df_test, threshold_date=None, xlabel=None
, ylabel=None, title=None):
    """ Plota os dataframes de teste e de treino com indicação visual do ponto de divisão

        Parametros
        ----------
        df_train: dataframe de treino
        df_test: dataframe de teste
        threshold_date: data de separação do dataframe
        xlabel: nome do eixo x do gráfico
        ylabel: nome do eixo y do gráfico
    """
    threshold_date = pd.to_datetime(threshold_date)
    parameters = {'ytick.labelsize': 15, 'xtick.labelsize': 15, 'axes.titlesize': 12}
    plt.rcParams.update(parameters)
    # rows, cols = df_train.shape
    if title is None:
        title = 'Dependent Variable'
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.lineplot(x='ds', y='y', label='y_train', data=df_train, ax=ax)
    sns.lineplot(x='ds', y='y', label='y_test', data=df_test, ax=ax)
    ax.axvline(threshold_date, color=sns_c[3], linestyle='--', label='train test split')
    ax.legend(loc='upper right')
    ax.set(title=title)
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]
    if xlabel is None:
        xlabel = 'Years'
    if ylabel is None:
        ylabel = 'Nº of Itens'

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    utls.save_figure(imgpath, 'train_test_split_', fig, isfigure=True)


def plot_seasonality(model=None, name=None, filename=None):
    """ Plot prophet seasonality

        Parametros
        -----------
        model: modelo Prophet
        name: nome da sazonalidade (weekly, daily, yearly, hourly)
    """
    if filename is None:
        filename = 'seasonality_'
    if model is None or name is None:
        raise Exception('Must inform model and name to plot seasonality!')
    else:
        if name == 'yearly':
            filename = filename + name + '_'
            fig = plt.figure(facecolor='w', figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
            ax.xaxis.set_major_locator(months)
            ax.set_xlabel('Day of year')
            ax.set_ylabel(name)
            fig = plt_seas(model, name=name, ax=ax, figsize=(12, 8))

        fig = plt_seas(model, name=name, figsize=(12, 8))
        utls.save_figure(imgpath, filename, plt)



def plot_trend(df, model, with_changepoints=False, filename=None):
    plt.figure(figsize=(12, 12))
    if filename is None:
        filename = 'trend_'
    if model is None:
        raise Exception('Must inform model to plot the trend!')
    else:
        x = df.ds.values
        y = df.trend.values
        cp = model.changepoints
    if with_changepoints:
        plt.plot(x, y)
        ymin, ymax = plt.ylim()
        plt.vlines(cp.values, ymin, ymax, linestyles="dashed")
        utls.save_figure(imgpath, filename, plt)
    else:
        plt.plot(x, y)
        utls.save_figure(imgpath, filename, plt)


def plot_TS_components(df, df_train, threshold_date):
    threshold_date = pd.to_datetime(threshold_date)
    mask = df['ds'] < threshold_date
    db = gendf.Database()
    df = db.prep_dataframe_components(df)
    try:
        decomposition_obj = seasonal_decompose(
            x=df_train.set_index('ds'),
            model='additive')
    except ValueError:
        decomposition_obj = seasonal_decompose(
            x=df_train.set_index('ds'),
            model='additive', period=12)
    fig, ax = plt.subplots(4, 1, figsize=(16, 16))
    # Observed time series.
    decomposition_obj.observed.plot(ax=ax[0])
    ax[0].set(title='observed')
    # Trend component.
    decomposition_obj.trend.plot(label='fit', ax=ax[1])
    df[mask][['ds', 'trend']].set_index('ds').plot(c=sns_c[1], ax=ax[1])
    ax[1].legend(loc='lower right')
    ax[1].set(title='trend')
    # Seasonal component.
    decomposition_obj.seasonal.plot(label='fit', ax=ax[2])
    df.assign(seasonal=lambda x: x['yearly_seas'] + x['monthly_seas'] + x['end_of_year']) \
        [mask][['ds', 'seasonal']] \
        .set_index('ds') \
        .plot(c=sns_c[2], ax=ax[2])
    ax[2].legend(loc='lower right')
    ax[2].set(title='seasonal')
    # Residual.
    decomposition_obj.resid.plot(label='fit', ax=ax[3])
    df[mask][['ds', 'noise']].set_index('ds').plot(c=sns_c[3], ax=ax[3])
    ax[3].legend(loc='lower right')
    ax[3].set(title='residual')

    fig.suptitle('Time Series Decomposition', y=1.01)
    for ext in ['png', 'jpeg', 'pdf']:
        plt.savefig(imgpath + '/' + f'daily_seasonality_components_comparison.{ext}', dpi=200)
    plt.tight_layout()
    return decomposition_obj


def plot_fcst_components(fcst, model, filename):
    plt = model.plot_components(fcst, figsize=(12,12))
    utls.save_figure(imgpath, filename, plt)


def plot_df_components(df):
    decompose_result = seasonal_decompose(df, period=12)
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid
    decompose_result.plot()


def plot_TS_grouped_by_day(filename, year, store):
    undesired_column = 'Unnamed: 0'
    df = pd.read_csv(csvpath + filename, sep=',')
    df['ds'] = pd.DatetimeIndex(df['ds'])
    df.set_index('ds', inplace=True)
    series_grouped_by_day = df['y'].resample('D').sum()
    df_grp_day = series_grouped_by_day.to_frame()
    df_grp_day.reset_index(level=0, inplace=True)
    df_grp_day['year'] = [d.year for d in df_grp_day.ds]
    df = df_grp_day.query('year >= '+year)
    df = df[df['y'] > 0]
    df_train, df_test, threshold_date = dtexp.split_dataframe(df)
    df_test.reset_index()
    predict_period = len(pd.date_range(threshold_date,max(df.index)))
    if undesired_column in df.columns:
        df.drop(columns=undesired_column, inplace=True)
    m=Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=predict_period)
    forecast = m.predict(future)
    m.plot(forecast, figsize=(8,6))
    plt.xlabel('Date',fontsize=12,fontweight='bold',color='gray')
    plt.ylabel('Product Sold',fontsize=12,fontweight='bold',color='gray')
    utls.save_figure(imgpath, store+'_fcst_grouped_by_day_', plt)

def plot_ts_autocorr(df):
    plot_acf(df)


def detrend_by_lr(df):
    least_squares = OLS(df["y"].values, list(range(df.shape[0])))
    result = least_squares.fit()
    fit = pd.Series(result.predict(list(range(df.shape[0]))), index=df.index)
    df_detrended = df["y"] - fit
    ax1 = plt.subplot(121)
    df_detrended.plot(figsize=(12, 4), color="tab:red", title="Linear Regression Fit", ax=ax1)
    ax2 = plt.subplot(122)
    df.plot(figsize=(12, 4), color="tab:red", title="Original Values", ax=ax2)
    decompose_result = seasonal_decompose(df_detrended.dropna())
    decompose_result.plot()


def plot_nr_of_y_per_year(df, ytitle='y', ci=95):
    df_new = dtexp.extract_date_features(df)
    fig, ax = plt.subplots(figsize=(14, 5))
    palette = sns.color_palette("mako_r", 4)
    a = sns.barplot(x="mes", y="y", hue='ano', data=df_new, ci=ci)
    a.set_title("Nr. of " + ytitle + " per year", fontsize=15)
    plt.legend(loc='upper right')
    utls.save_figure(imgpath, 'nof_y_per_year_', plt)
    # plt.show()
    return plt


def plot_negative_values(df, filename=None, by_hour=False, by_day=False, by_month=False, by_year=False, by_store=False):
    if filename is None:
        filename = 'plot_negative_values_'
    negative_df = df[df['y'] < 0]
    if negative_df.empty:
        msg = 'There is no negative values for y column in this dataframe'
        return msg
    else:
        negative_df['ds'] = pd.to_datetime(negative_df['ds'])
        if by_hour:
            filename = filename + 'hour_'
            try:
                ax = negative_df['ds'].dt.hour.value_counts().plot(kind='barh', ylabel='Hour',
                                                                   title='Nº of negative values by Hour')
                utls.save_figure(neg_values_path, filename, ax, isfigure=True, is_ax=True)
            except IndexError:
                print('There is no negative values')
        elif by_day:
            filename = filename + 'day_'
            ax = negative_df['ds'].dt.strftime('%A').value_counts().plot(kind='barh', ylabel='Weekday',
                                                                         title='Nº of negative values by Weekday')
            utls.save_figure(neg_values_path, filename, ax, isfigure=True, is_ax=True)
        elif by_month:
            filename = filename + 'month_'
            ax = negative_df['ds'].dt.strftime('%B').value_counts().plot(kind='barh', ylabel='Month',
                                                                         title='Nº of negative values by Month')
            utls.save_figure(neg_values_path, filename, ax, isfigure=True, is_ax=True)
        elif by_year:
            filename = filename + 'year_'
            ax = negative_df['ds'].dt.strftime('%Y').value_counts().plot(kind='barh', ylabel='Year',
                                                                         title='Nº of negative values by Year')
            utls.save_figure(neg_values_path, filename, ax, isfigure=True, is_ax=True)
        elif by_store:
            filename = filename + 'store_'
            ax = negative_df[negative_df['y']<0].groupby('store')['y'].count().nlargest(10).plot(kind='barh', ylabel='Negative values',
                                                                         title='Top 10 stores with negative values')
            utls.save_figure(neg_values_path, filename, ax, isfigure=True, is_ax=True)


def plot_negative_values_test(df, filename=None):
    if filename is None:
        filename = 'plot_negative_values_'
    negative_df = df[df['y'] < 0]
    if negative_df.empty:
        msg = 'There is no negative values for y column in this dataframe'
        return msg
    else:
        negative_df['ds'] = pd.to_datetime(negative_df['ds'])
        filename = filename + 'hour_'
        try:
            plt = negative_df['ds'].dt.hour.value_counts().plot(kind='barh', ylabel='Hour',
                                                                title='Nº of negative values by Hour')
            utls.save_figure(imgpath, filename, plt, is_ax=True)
            filename = filename + 'day_'
            plt = negative_df['ds'].dt.strftime('%A').value_counts().plot(kind='barh', ylabel='Weekday',
                                                                          title='Nº of negative values by Weekday')
            utls.save_figure(imgpath, filename, plt, is_ax=True)
            filename = filename + 'month_'
            plt = negative_df['ds'].dt.strftime('%B').value_counts().plot(kind='barh', ylabel='Month',
                                                                          title='Nº of negative values by Month')
            utls.save_figure(imgpath, filename, plt, is_ax=True)
            filename = filename + 'year_'
            plt = negative_df['ds'].dt.strftime('%Y').value_counts().plot(kind='barh', ylabel='Year',
                                                                          title='Nº of negative values by Year')
            utls.save_figure(imgpath, filename, plt, is_ax=True)
        except IndexError:
            print('There is no negative values')


def plot_heatmap(df
                 , filename=None
                 , store=0
                 , highlight_month=None
                 , year=None
                 , size=(12, 10)
                 , cellvalues=None
                 , filler=None):
    columns = ['ds', 'y']
    custom_colors = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#fefae0", "#f3d5b5", "#e7bc91", "#d4a276"
                                                                             , "#bc8a5f", "#a47148", "#8b5e34"
                                                                             , "#6f4518", "#603808", "#583101"])
    if filename is None:
        filename = 'heatmap_'+str(store)
    if columns[0] in list(df.columns) and columns[1] in list(df.columns):
        if len(df['ds'].dt.strftime('%Y').unique().tolist()) < 5:
            size = None
        df.reset_index(inplace=True)
        df.set_index('ds', inplace=True)
        if year is None:
            fig, ax = calplot.calplot(df['y']
                                      , suptitle="Store " + str(store) + " Heatmap"
                                      , edgecolor=highlight_month
                                      , figsize=size
                                      , cmap=custom_colors)
        else:
            fig, ax = calplot.calplot(df.loc[year, 'y']
                                      , suptitle="Store " + str(store) + " Heatmap"
                                      , edgecolor=highlight_month
                                      , figsize=size
                                      , textformat=cellvalues
                                      , textfiller=filler
                                      , cmap=custom_colors)

        utls.save_figure(imgpath, filename, fig, isfigure=True, is_ax=True)


def zoom_ts_plot(df):
    df_neg = df[df['y'] < 0]
    if df_neg.shape[0] != 0:
        df_neg.set_index('ds', inplace=True)
        df_orig = df.copy()
        df_orig.set_index('ds', inplace=True)
        # Plot
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.plot(df_orig)

        # Label the axis
        ax.set_xlabel('ds')
        ax.set_ylabel('y')

        # I want to select the x-range for the zoomed region. I have figured it out suitable values
        # by trial and error. How can I pass more elegantly the dates as something like
        x1 = df_neg.index[0]
        x2 = df_neg.index[len(df_neg.index) - 1]

        # select y-range for zoomed region
        y1 = df_neg.y[0]
        y2 = df_neg.y[len(df_neg.y) - 1]

        # Make the zoom-in plot:
        axins = zoomed_inset_axes(ax, 2, loc=1)  # zoom = 2
        axins.plot(df_orig)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="0.1", ec="0.5")
        plt.draw()
        plt.show()
    else:
        print('Forecast has no negative prediction values')


def highlight_ts_dots(df, threshold_values=0, is_forecast=False, title=None, color='black'):
    if is_forecast:
        query = 'yhat '
        plt.scatter(df.ds, df.yhat, alpha=0.5)
    else:
        query = 'y '
        plt.scatter(df.ds, df.y, alpha=0.5)
    if threshold_values <= 0:
        df_dots = df.query(query + '<= '+str(threshold_values))
    else:
        df_dots = df.query(query + '> ' + str(threshold_values))

    # set x-axis label and specific size
    plt.xlabel('Date', size=16)
    # set y-axis label and specific size
    plt.ylabel('Sales', size=16)
    if title is None:
        title = 'Viewing negative datapoints on Time Series'
    # set plot title with specific size
    plt.title(title, size=16)
    title = title.replace(' ', '_') + '_'
    if is_forecast:
        plt.scatter(df_dots.ds, df_dots.yhat, c=color)
        utls.save_figure(imgpath, title, plt)
    else:
        plt.scatter(df_dots.ds, df_dots.y, c=color)
        utls.save_figure(imgpath, title, plt)



def view_df_hours_distribution(df,store_no):
    file_name = store_no+'_working_hour.csv'
    freq_hours = make_ds.count_values_per_hour(df, filename=file_name)
    df_distrib_horas= make_ds.load_processed_dataset(file_name)
    df_distrib_horas.columns
    df_distrib_horas.rename(columns={'ds.1': 'y'}, inplace=True)
    hora=df_distrib_horas['ds'].tolist()
    qtd_units=df_distrib_horas['y'].tolist()
    x_values=hora
    y_values=qtd_units
    title=store_no+' Store ' +' - Hour with values distributed throughout the day'
    xaxes_title='Hours'
    yaxis_title='Nº of data with hour'
    plot_results(x_values,y_values,title=title,xaxes_title=xaxes_title,yaxis_title=yaxis_title)

    fig, ax = plt.subplots(figsize=(14, 5))
    palette = sns.color_palette("mako_r", 4)
    a = sns.barplot(x="ds", y="y", data=df_distrib_horas, ci=95)
    a.set_title(title, fontsize=15)
    plt.legend(loc='upper right')
    utls.save_figure(imgpath, file_name[:-3], plt)
    utls.save_figure(imgpath, file_name[:-3], fig, isfigure=True)
    return freq_hours


def view_weekly_sales_dist(df, store_no):
    df_extracted_features = dtexp.extract_date_features(df)
    week_day = pd.DataFrame(df_extracted_features.groupby("diasemana")["y"].sum()).reset_index().sort_values('diasemana')
    week_day.reset_index(level=0, inplace=True)
    diasemana=week_day.diasemana
    week_day.rename(columns={'y': 'Products Sales'}, inplace=True)
    prodsmedians=week_day.groupby('diasemana')['Products Sales'].median()

    title='Store '+store_no +' Days of week sales distribution'
    file_name = title.replace(' ','_')
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(figsize=(14, 5))

    ax = sns.barplot(x=diasemana,y='Products Sales',data=week_day, palette="ch:.75")
    ax.set_title(title, fontsize=15)
    plt.legend(loc='upper right')
    for i in range(len(prodsmedians)):
        ax.annotate(str(prodsmedians[i]), xy= (i, prodsmedians[i]), ha='center', fontsize=15)
    utls.save_figure(imgpath, file_name[:-3], plt)
    utls.save_figure(imgpath, file_name[:-3], fig, isfigure=True)


def plot_data_hourly_distributed(df, store_no):
    undesired_column = 'Unnamed: 0'
    df.reset_index(level=0,inplace=True)
    df['ds'] = pd.DatetimeIndex(df['ds'])
    if undesired_column in df.columns:
        df_store_63.drop(columns=undesired_column, inplace=True)
    df[df['y'] > 0].groupby(df['ds'].dt.hour)['ds'].count()
    most_freq_hours = dtview.view_df_hours_distribution(df, store_no)
    valid_hours = pd.DataFrame(df[df['y'] > 0].groupby(df['ds'].dt.hour)['ds'].count())
    valid_hours.index.name='hour'


#creating the animation function.
def animate(i, df, ax1):

    lines =  df.iloc[0:int(i+500)] #   set the variable data to contain 0 to the (i+1)th row. I wanted only the last data and the predictions to be animated, so put i+500.

    xs = []
    ys = []
    zs = []
    for line in lines:
        if len(line)>1:
            xs = lines['ds']
            ys = lines['yhat']
#the predictions
            zs = lines['y']
#the normal data point

# Creating the ax labels, title, legend and plotting value
    ax1.clear()
    ax1.set(title='Predicting the future traffic of my website',xlabel= "Time", ylabel='Traffic')
    ax1.axis(xmin= (df['ds'].min()), xmax=(df['ds'].max()))
    ax1.axis(ymin= (df['y'].min()-1), ymax=(df['y'].max()+5)) #adding a bit of margin with -1 and +5

    ax1.plot(xs, ys, label='prediction', color='#0693e3')
    ax1.plot(xs, zs, marker = '.' ,linewidth = 0.0, label='data', color='#afdedc') #linewidth is not visible at 0.1, I change to 0.3
    ax1.legend(loc='upper left', frameon=True, framealpha = 0.5 )
