import pandas as pd
import sqlalchemy as dbase
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column, Integer, String, Float, DateTime
from src.data import prepare_dataframe
import numpy as np
import settings as settings
from uuid import uuid4
import dask.dataframe as dd
import timeit
import datetime
import logging

path, csvpath, imgpath, neg_values_path, large_file_path, model_path = settings.get_file_path()
DBNAME, DBUSER, DBPASS, DBHOST, DBPORT = settings.get_acces_db_data()
logging.basicConfig(filename="FORECAST.log",
                    level=logging.DEBUG,
                    filemode="w",
                    format=settings.settings.get('LOGFORMAT'),
                    datefmt=settings.settings.get('LOGDATEFORMAT'), )


class Database(object):

    def __init__(self):
        DATABASES = {
            'prophetdatasets': {
                'NAME': DBNAME,
                'USER': DBUSER,
                'PASSWORD': DBPASS,
                'HOST': DBHOST,
                'PORT': DBPORT,
            },
        }
        self.dbase = dbase
        # # choose the database to use
        self.db = DATABASES['prophetdatasets']
        # # construct an engine connection string
        self.engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
            user=self.db['USER'],
            password=self.db['PASSWORD'],
            host=self.db['HOST'],
            port=self.db['PORT'],
            database=self.db['NAME'],
        )
        # # create sqlalchemy engine
        # # worth the test engine.execution_options(stream_results=True) for read as chunk (lower memory, same speed)
        self.engine = self.dbase.create_engine(self.engine_string)
        self.metadata = self.dbase.MetaData()
        logging.debug("Creating Engine at :".format(datetime.datetime.now()))

    def get_dataframe(self, table=None, cols=None, size=None, col_to_rename=None, query=False):
        # # read a table from database into pandas dataframe
        start = timeit.default_timer()
        if not query:
            df = pd.read_sql_table(table, self.engine, columns=cols, chunksize=size)
            if col_to_rename is not None:
                df.rename(columns={col_to_rename: 'y'}, inplace=True)
            logging.debug("Pandas reading table time: %s", timeit.default_timer() - start)
            logging.debug("\n ".format(datetime.datetime.now()))
            # If query is true, then get all stores ids and return as dataframe to perform over it
        else:
            conn = self.engine.connect()
            tablename = self.dbase.Table(table, self.metadata, autoload=True, autoload_with=self.engine)
            query = self.dbase.sql.select([tablename.columns.store.distinct()])
            result_proxy = conn.execute(query)
            result_set = result_proxy.fetchall()
            df = pd.DataFrame(result_set)
            df.columns = result_set[0].keys()
            logging.debug("Pandas reading table time: %s", timeit.default_timer() - start)
        return df

    def get_df_with_dask(self, filename):
        csv_file = csvpath + '/' + filename
        df = dd.read_csv(csv_file)
        return df

    def get_dataset_to_forecast(self, tblname=None):
        try:
            if tblname[-3:] == 'csv':
                df_csv = pd.read_csv(csvpath + '/' + tblname)
                print(df_csv)
                df_csv['datetime'] = df_csv["datetime"].astype("M8[us]")
                df_hourly_cli_prepared = prepare_dataframe.get_dataset(df_csv, ishourly=True, isclient=True)
                df_daily_cli_prepared = prepare_dataframe.get_dataset(df_csv, ishourly=False, isclient=True)
            else:
                df = self.get_dataframe(tblname)
                df_hourly_cli_prepared = prepare_dataframe.get_dataset(df, ishourly=True, isclient=True)
                df_daily_cli_prepared = prepare_dataframe.get_dataset(df, ishourly=False, isclient=True)
        except:
            pass

        return df_hourly_cli_prepared, df_daily_cli_prepared

    def get_negative_dataset_from_csv(self, filename=None):
        try:
            df_csv = pd.read_csv(csvpath + '/' + filename)
            df_result = prepare_dataframe.get_dataset(df_csv, is_negative_values=True)
        except:
            print("Error! Returning empty dataframe")
            df_result = pd.DataFrame()

        return df_result

    # # In construction
    def prepare_csv_to_database(self, ):
        pass

    def prep_dataframe_components(self, df):
        # Get date variables.
        df['day_of_month'] = df['ds'].dt.day
        df['month'] = df['ds'].dt.month
        df['daysinmonth'] = df['ds'].dt.daysinmonth
        df['week'] = df['ds'].dt.week

        # Time Series Components
        ## Trend
        df['trend'] = np.power(df.index.values + 1, 2 / 5) + np.log(df.index.values + 3)
        ## Seasonal
        df['monthly_seas'] = np.cos(2 * np.pi * df['day_of_month'] / df['daysinmonth'])
        df['yearly_seas'] = 1.2 * (np.sin(np.pi * df['month'] / 3) + np.cos(2 * np.pi * df['month'] / 4))
        df['end_of_year'] = - 8.5 * np.exp(- ((df['week'] - 51.5) / 1.0) ** 2) \
            ## Gaussian noise
        df['noise'] = np.random.normal(loc=0.0, scale=0.3, size=df.shape[0])

        # Target variable.
        df['y'] = df['trend'] \
                  + df['monthly_seas'] \
                  + df['yearly_seas'] \
                  + df['end_of_year'] \
                  + df['noise']

        return df

    def insert_forecastability(self, store, size, qtd_negative_values, ApEn, SampEn, m, r):
        if self.table_exists('entropy_analysis'):
            entropy_table = self.dbase.Table('entropy_analysis', self.metadata, autoload=True,
                                             autoload_with=self.engine)
            self.engine.execute(entropy_table.insert()
                                , analysis_moment=datetime.datetime.now()
                                , store=store
                                , df_size=size
                                , qtd_negative_values=qtd_negative_values
                                , ApEn=ApEn
                                , SampEn=SampEn
                                , m=m
                                , r=r)

    def insert_model_performance(self, store, type, horizon, mse, rmse, mae, mape, mdape, coverage, params):
        if not self.table_exists('cross_validation'):
            self.create_table_cross_validation()
        cross_validation = self.dbase.Table('cross_validation', self.metadata, autoload=True
                                            , autoload_with=self.engine)
        self.engine.execute(cross_validation.insert()
                            , analysis_moment=datetime.datetime.now()
                            , store=store
                            , type=type
                            , horizon=horizon
                            , mse=mse
                            , rmse=rmse
                            , mae=mae
                            , mape=mape
                            , mdape=mdape
                            , coverage=coverage
                            , params=params)


    def insert_manual_evaluation(self, store, horizon, rmse, mae, mape
    , seasonality_prior_scale
    , holidays_prior_scale
    , changepoint_prior_scale
    , outlier):
        if not self.table_exists('manual_evaluation'):
            self.create_table_manual_evaluation()
        manual_evaluation = self.dbase.Table('manual_evaluation', self.metadata, autoload=True
                                            , autoload_with=self.engine)
        self.engine.execute(manual_evaluation.insert()
                            , analysis_moment=datetime.datetime.now()
                            , store=store
                            , horizon=horizon
                            , rmse=rmse
                            , mae=mae
                            , mape=mape
                            , seasonality_prior_scale=seasonality_prior_scale
                            , holidays_prior_scale=holidays_prior_scale
                            , changepoint_prior_scale=changepoint_prior_scale
                            , outlier=outlier)

    # # In construction
    def insert_model_params(self
                            , store
                            , growth
                            , changepoint_range
                            , seasonality_mode
                            , yearly_fourier
                            , yearly_prior_scale
                            , weekly_fourier
                            , weekly_prior_scale
                            , daily_fourier
                            , daily_prior_scale
                            , hourly_fourier
                            , hourly_prior_scale
                            , seasonality_prior_scale
                            , holidays_prior_scale
                            , changepoint_prior_scale
                            , mcmc_samples
                            , interval_width
                            , uncertainty_samples
                            , forecast_analysis_id_seq):
        if not self.table_exists('prophet_model_params'):
            self.create_table_prophet_model_params()
        prophet_model_params = self.dbase.Table('prophet_model_params', self.metadata, autoload=True
                                                , autoload_with=self.engine)
        self.engine.execute(prophet_model_params.insert()
                            , moment=datetime.datetime.now()
                            , store=store
                            , growth=growth
                            , changepoint_range=changepoint_range
                            , seasonality_mode=seasonality_mode
                            , yearly_fourier=yearly_fourier
                            , yearly_prior_scale=yearly_prior_scale
                            , weekly_fourier=weekly_fourier
                            , weekly_prior_scale=weekly_prior_scale
                            , daily_fourier=daily_fourier
                            , daily_prior_scale=daily_prior_scale
                            , hourly_fourier=hourly_fourier
                            , hourly_prior_scale=hourly_prior_scale
                            , seasonality_prior_scale=seasonality_prior_scale
                            , holidays_prior_scale=holidays_prior_scale
                            , changepoint_prior_scale=changepoint_prior_scale
                            , mcmc_samples=mcmc_samples
                            , interval_width=interval_width
                            , uncertainty_samples=uncertainty_samples
                            , forecast_analysis_id_seq=forecast_analysis_id_seq
                            , mae=mae
                            , mape=mape)


    def insert_forecast_analysis(self, store, ApEn, df_size
        , fcst_period, freq_hours, work_hours, mse_outlier, rmse_outlier
        , mae_outlier, mape_outlier, mse_without_outlier, rmse_without_outlier
        , mae_without_outlier, mape_without_outlier, model_outlier_file_name
        , model_without_outlier_file_name):
        if not self.table_exists('forecast_analysis'):
            self.create_table_forecast_analysis()
        forecast_analysis = self.dbase.Table('forecast_analysis', self.metadata, autoload=True
                                            , autoload_with=self.engine)
        self.engine.execute(forecast_analysis.insert()
                            , analysis_moment=datetime.datetime.now()
                            , store=store
                            , ApEn=ApEn
                            , df_size=df_size
                            , fcst_period=fcst_period
                            , freq_hours=freq_hours
                            , work_hours=work_hours
                            , mse_outlier=mse_outlier
                            , rmse_outlier=rmse_outlier
                            , mae_outlier=mae_outlier
                            , mape_outlier=mape_outlier
                            , mse_without_outlier=mse_without_outlier
                            , rmse_without_outlier=rmse_without_outlier
                            , mae_without_outlier=mae_without_outlier
                            , mape_without_outlier=mape_without_outlier
                            , model_outlier_file_name=model_outlier_file_name
                            , model_without_outlier_file_name=model_without_outlier_file_name)


    def table_exists(self, name):
        ins = self.dbase.inspect(self.engine)
        result = ins.dialect.has_table(self.engine.connect(), name)
        logging.debug('Table "{}" exists: {}'.format(name, result))
        return result

    def create_table_entropy_analysis(self):
        id_seq = dbase.Sequence('entropy_analysis_id_seq', self.metadata)
        entropy_analysis = self.dbase.Table('entropy_analysis'
                                            , self.metadata
                                            , dbase.Column('id_seq', dbase.Integer, primary_key=True)
                                            , dbase.Column('analysis_moment', dbase.DateTime, nullable=False)
                                            , dbase.Column('store', dbase.Integer, nullable=False)
                                            , dbase.Column('df_size', dbase.Integer, nullable=False)
                                            , dbase.Column('qtd_negative_values', dbase.Integer, nullable=False)
                                            , dbase.Column('ApEn', dbase.Float, nullable=False)
                                            , dbase.Column('SampEn', dbase.Float, nullable=False)
                                            , dbase.Column('m', dbase.Integer, nullable=False)
                                            , dbase.Column('r', dbase.Float, nullable=False)
                                            )
        entropy_analysis.create(self.engine)

    def create_table_cross_validation(self):
        id_seq = dbase.Sequence('cross_validation_id_seq', self.metadata)
        cross_validation = self.dbase.Table('cross_validation'
                                            , self.metadata
                                            , dbase.Column('id_seq', dbase.Integer, primary_key=True)
                                            , dbase.Column('analysis_moment', dbase.DateTime, nullable=False)
                                            , dbase.Column('store', dbase.Integer, nullable=False)
                                            , dbase.Column('type', dbase.Integer, nullable=False)
                                            , dbase.Column('horizon', dbase.TEXT, nullable=False)
                                            , dbase.Column('mse', dbase.Float, nullable=False)
                                            , dbase.Column('rmse', dbase.Float, nullable=False)
                                            , dbase.Column('mae', dbase.Float, nullable=False)
                                            , dbase.Column('mape', dbase.Float, nullable=False)
                                            , dbase.Column('mdape', dbase.Float, nullable=False)
                                            , dbase.Column('coverage', dbase.Float, nullable=False)
                                            , dbase.Column('params', dbase.TEXT, nullable=False)
                                            )
        cross_validation.create(self.engine)


    def create_table_manual_evaluation(self):
        id_seq = dbase.Sequence('manual_evaluation_id_seq', self.metadata)
        manual_evaluation = self.dbase.Table('manual_evaluation'
                                            , self.metadata
                                            , dbase.Column('id_seq', dbase.Integer, primary_key=True)
                                            , dbase.Column('analysis_moment', dbase.DateTime, nullable=False)
                                            , dbase.Column('store', dbase.Integer, nullable=False)
                                            , dbase.Column('horizon', dbase.TEXT, nullable=False)
                                            , dbase.Column('rmse', dbase.Float, nullable=False)
                                            , dbase.Column('mae', dbase.Float, nullable=False)
                                            , dbase.Column('mape', dbase.Float, nullable=False)
                                            , dbase.Column('seasonality_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('holidays_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('changepoint_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('outlier', dbase.Integer, nullable=False)
                                            )
        manual_evaluation.create(self.engine)


    def create_table_prophet_model_params(self):
        id_seq = dbase.Sequence('prophet_model_params', self.metadata)
        prophet_model_params = self.dbase.Table('prophet_model_params'
                                            , self.metadata
                                            , dbase.Column('id_seq', dbase.Integer, primary_key=True)
                                            , dbase.Column('moment', dbase.DateTime, nullable=False)
                                            , dbase.Column('store', dbase.Integer, nullable=False)
                                            , dbase.Column('growth', dbase.TEXT, nullable=False)
                                            , dbase.Column('changepoint_range', dbase.Float, nullable=False)
                                            , dbase.Column('seasonality_mode', dbase.TEXT, nullable=False)
                                            , dbase.Column('yearly_fourier', dbase.Float, nullable=False)
                                            , dbase.Column('yearly_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('weekly_fourier', dbase.Float, nullable=False)
                                            , dbase.Column('weekly_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('daily_fourier', dbase.Float, nullable=False)
                                            , dbase.Column('daily_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('hourly_fourier', dbase.Float, nullable=False)
                                            , dbase.Column('hourly_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('seasonality_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('holidays_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('changepoint_prior_scale', dbase.Float, nullable=False)
                                            , dbase.Column('mcmc_samples', dbase.Integer, nullable=False)
                                            , dbase.Column('interval_width', dbase.Float, nullable=False)
                                            , dbase.Column('uncertainty_samples', dbase.Integer, nullable=False)
                                            , dbase.Column('forecast_analysis_id_seq', dbase.Integer, nullable=False)
                                            )
        prophet_model_params.create(self.engine)


    def create_table_forecast_analysis(self):
        id_seq = dbase.Sequence('forecast_analysis_id_seq', self.metadata)
        entropy_analysis = self.dbase.Table('forecast_analysis'
                                            , self.metadata
                                            , dbase.Column('id_seq', dbase.Integer, primary_key=True)
                                            , dbase.Column('analysis_moment', dbase.DateTime, nullable=False)
                                            , dbase.Column('store', dbase.Integer, nullable=False)
                                            , dbase.Column('df_size', dbase.Integer, nullable=False)
                                            , dbase.Column('ApEn', dbase.Float, nullable=False)
                                            , dbase.Column('fcst_period', dbase.TEXT, nullable=False)
                                            , dbase.Column('freq_hours', dbase.TEXT, nullable=False)
                                            , dbase.Column('work_hours', dbase.TEXT, nullable=False)
                                            , dbase.Column('mse_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('rmse_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('mae_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('mape_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('mse_without_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('rmse_without_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('mae_without_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('mape_without_outlier', dbase.Float, nullable=False)
                                            , dbase.Column('model_outlier_file_name', dbase.TEXT, nullable=False)
                                            , dbase.Column('model_without_outlier_file_name', dbase.TEXT, nullable=False)
                                            )
        entropy_analysis.create(self.engine)


    def get_forecast_analysis_table(self):
        meta = MetaData()
        fcst_analysis = Table('forecast_analysis', meta
                  , Column('analysis_moment', DateTime)
                  , Column('store', Integer)
                  , Column('df_size', Integer)
                  , Column('ApEn', Float)
                  , Column('fcst_period', String)
                  , Column('freq_hours', String)
                  , Column('work_hours', String)
                  , Column('mse_outlier', Float)
                  , Column('rmse_outlier', Float)
                  , Column('mae_outlier', Float)
                  , Column('mape_outlier', Float)
                  , Column('mse_without_outlier', Float)
                  , Column('rmse_without_outlier', Float)
                  , Column('mae_without_outlier', Float)
                  , Column('mape_without_outlier', Float)
                  , Column('model_outlier_file_name', String)
                  , Column('model_without_outlier_file_name', String)
                  )
        return fcst_analysis
