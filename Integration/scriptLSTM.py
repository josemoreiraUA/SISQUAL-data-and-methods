import datetime, os
import pandas as pd
import numpy as np
from numpy import split, array, zeros
import pyodbc
import settings as settings

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Bidirectional, Dropout


def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f'{date_text} is in incorrect format! Please use yyyy-mm-dd.')

def get_schedule(store, start, end):
    q_start = start.replace('-','')
    q_end = end.replace('-','')

    query_schedule = f"""
    WITH H (employeecode, date, ScheduleCode)
    AS
    (
       select distinct employeecode, date, ScheduleCode
       from processblock
       where employeecode between 8000001 and 8999999
       and date between '{q_start}' and '{q_end}'
    )


    SELECT RosterCode=L.CodLocal,
       [Date]=H.date,
       --[Schedule]=S.Description,
       --[Start]=[dbo].[GetScheduleStartDate](H.date, S.MaxProDay1StartMinute, S.MaxProDay1EndMinute, S.MaxProDay2StartMinute, S.MaxProDay2EndMinute),
       --[End]=[dbo].[GetScheduleEndDate](H.date,  S.MaxProDay1StartMinute, S.MaxProDay1EndMinute, S.MaxProDay2StartMinute, S.MaxProDay2EndMinute),
       [StartEstimation]=[dbo].[GetScheduleStartDate](H.date, S.ClockingsDay1StartMinute, S.ClockingsDay1EndMinute, S.ClockingsDay2StartMinute, S.ClockingsDay2EndMinute),
       [EndEstimation]=[dbo].[GetScheduleEndDate](H.date, S.ClockingsDay1StartMinute, S.ClockingsDay1EndMinute, S.ClockingsDay2StartMinute, S.ClockingsDay2EndMinute)
    FROM H
    LEFT JOIN Schedule S ON H.ScheduleCode=S.Code
    LEFT JOIN mpLocal L ON L.CodEmpregadoHorarioLocalMes=H.employeecode
    WHERE L.CodLocal = '{store}'
    ORDER BY 1,2
    """

    # Access databases
    SERVER, DATABASE, DATABASE_SCHEDULE, USERNAME, PASSWORD = settings.get_access_db_data()
    USERNAME = 'sa'    # no idea why this is needed!
    con_schedule = pyodbc.connect(driver='ODBC Driver 17 for SQL Server', server=SERVER, database=DATABASE_SCHEDULE, uid=USERNAME, pwd=PASSWORD)
    schedule_df = pd.read_sql(query_schedule, con_schedule)

    return schedule_df

def get_trainingData(store, start, end, period=60, target='sales'):
    # Choose between sale value or number of clients
    if target=='clients':
        q = 'SUM([tickets]) AS [y]'
    else:
        q = 'ROUND(SUM([cantidad]), 2) AS [y]'

    query = f"""
    SELECT DATEADD(minute, -(datepart(MI, [fechaHoraInicio]) % '{period}'), [fechaHoraInicio]) AS ds, {q}
    FROM [sisqualFORECASTDATA].[dbo].[tienda_ventas_caja]
    WHERE [tienda] = {store} AND [fechaHoraInicio] < '{start}'
    GROUP BY DATEADD(minute, -(datepart(MI, [fechaHoraInicio]) % '{period}'), [fechaHoraInicio])
    ORDER BY ds
    """
    # Access databases
    SERVER, DATABASE, DATABASE_SCHEDULE, USERNAME, PASSWORD = settings.get_access_db_data()
    USERNAME = 'sa'    # no idea why this is needed!
    con = pyodbc.connect(driver='ODBC Driver 17 for SQL Server', server=SERVER, database=DATABASE, uid=USERNAME, pwd=PASSWORD)
    df = pd.read_sql(query, con)

    return df

def preprocessing(df, schedule_df):
    df.set_index('ds', inplace=True)
    # 1) find biggest schedule
    pre_schedule_df = schedule_df[schedule_df['StartEstimation'] != schedule_df['EndEstimation']]
    open_hour = str(pre_schedule_df['StartEstimation'].dt.hour.min())+':00:00'
    close_hour = str(pre_schedule_df['EndEstimation'].dt.hour.max())+':00:00'
    df.loc[df['y'] == 0] = -500    # preserve real '0's
    # 2) resample data to fill in blank spaces
    df = df.resample('H').sum()
    df.loc[df['y'] == 0] = np.nan
    df.loc[df['y'] < 0] = 0    # restore real '0's
    # 3) filter and input
    df = df.between_time(open_hour, close_hour)
    # get list of days
    for dt in np.unique(df.index.date):
        # input each day individualy
        df.loc[str(dt)] = df.loc[str(dt)].interpolate(method='linear', limit_direction='both')
    df.fillna(df.mean(), inplace = True)

    return df, open_hour, close_hour

def to_univariate(df, init_train_set):
    df = df.iloc[init_train_set:]
    df = df[['y']]
    # ensure all data is float
    values = df.values
    values = values.astype('float32')

    return values

def train_split(values, data_split):
    # split into train and test, leave the last test_set blocks for n_output timesteps
    train = values
    # Normalization
    scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1)) 
    train = scaler.fit_transform(train)
    # restructure into windows, for the sliding window method
    train = array(split(train, len(train) / data_split))

    return train, scaler

# train the model
def build_model(train, n_input, n_output, stride, units=32, epochs=50, batch_size=0):
    
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_output, stride)
    
    # Model variables
    verbose = 1 #batch_size 1 IS SGD, 1<BATCH_SIZE<SIZE IS MINIBATCH GD AND BATCH_SIZE=SIZE IS BATCH GD
    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
    
    # final data preparation for the model
    # reshape train_output into [samples, timesteps, features] for the LSTMs
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], n_features)
    print(train_x.shape, train_y.shape)
    # define model
    model = Sequential()
    
    # VANILLA LSTM
    model.add(LSTM(units, activation='tanh', input_shape=(n_timesteps, n_features)))
    model.add((Dense(n_output)))
    
    model.compile(loss="mse", optimizer='adam', metrics=["mae"])  #Reminder: LOSS function is MSE but others can be used!
    
    early = tensorflow.keras.callbacks.EarlyStopping('loss', patience=5)

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early]) #tensorboard_callback,
    print(model.summary())
    
    return model

# convert history into inputs and outputs - framing to a supervised learning problem
def to_supervised(train, n_input, n_output, stride=1):
    # flatten data
    data = train.reshape(train.shape[0] * train.shape[1], train.shape[2])
    train_x, train_y = list(), list()
    in_start = 0
    # step over the entire history one STRIDE step at a time
    for _ in range(0, len(data), stride):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_output
        # ensure we have enough data for this instance
        if out_end <= len(data):
            train_x.append(data[in_start:in_end, :])
            train_y.append(data[in_end:out_end, 0])
        # move along stride time steps
        in_start += stride
    return array(train_x), array(train_y)

# make the multi-step forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    
    # retrieve last n_input observations to predict with
    input_x = data[-n_input:, :]
    
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    
    # forecast the next n_output steps
    yhat = model.predict(input_x, verbose=0)
    
    # we only want the forecast variable
    yhat = yhat[0]
    return yhat

# invert the scaling
def invTransformTarget(scaler, data):
    dummy = pd.DataFrame(zeros((len(data), scaler.n_features_in_)))
    dummy[0] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=dummy.columns)
    return dummy[0].values

def postprocessing(schedule_df, fcst):
    pos_df = None

    for i in schedule_df.index:
        if str(schedule_df['StartEstimation'][i]) == str(schedule_df['EndEstimation'][i]):
            continue

        day = str(schedule_df['Date'][i].date())
        day_start = str(schedule_df['StartEstimation'][i].time())
        day_end = str(schedule_df['EndEstimation'][i].time())

        day_df = fcst.loc[day].between_time(day_start, day_end)

        if pos_df is None:
            pos_df = day_df.copy()
        else:
            pos_df = pd.concat([pos_df, day_df], ignore_index=False)

    return pos_df

def save_forecast(pos_df):
    # .. to do!
    pass



# args
store = 30
start = '2018-01-01'
end = '2018-01-31'
period = 60
target = 'sales'

# validate args ...
schedule_df = get_schedule(store, start, end)
df = get_trainingData(store, start, end, period, target)
df, open_hour, close_hour = preprocessing(df, schedule_df)

# PARAMETERS
N_WEEKS = 5
DAYS_OF_WEEK = 7
D = pd.to_datetime(close_hour) - pd.to_datetime(open_hour)
N_HOURS = int(D.total_seconds()/3600)+1

n_input = N_HOURS*2                                          # steps used to predict (autoregressive order) p
n_output = N_WEEKS*DAYS_OF_WEEK*N_HOURS               # steps to predict (forecast horizon) H
data_split = 1                                 # to split the data in windows
stride = 1                                        # stride value for the sliding window method (overlapped vs non-overlapped)
init_train_set = 0                          # refers to when the train_set starts, this is useful for the sliding window method

values = to_univariate(df, init_train_set)
train, scaler = train_split(values, data_split)

# history of windows, is updated for each prediction
history = [x for x in train]

# the model is trained and retrained for every number of n_output to predict
model = build_model(array(history), n_input, n_output, stride, epochs=5)  
train_size = len(history) # number of training windows
len_train = train_size*n_output # actual training size/length
            
# predict the next n_output steps
yhat_sequence = forecast(model, history, n_input)
    
# invert the scaling on predictions
yhat_sequence = invTransformTarget(scaler, yhat_sequence)

# rearange
times = pd.date_range(start=start, periods=24*DAYS_OF_WEEK*N_WEEKS, freq='H')
fcst = pd.DataFrame(index=times)
fcst = fcst.between_time(open_hour, close_hour)
fcst['yhat'] = yhat_sequence

pos_df = postprocessing(schedule_df, fcst)

print(pos_df)
