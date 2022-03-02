import numpy as np
import pandas as pd

from fast_forecaster.config import config
from fast_forecaster.pipeline import Pipe 
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import timedelta


def to_supervised(input_data, n_input_samples, n_output_samples, stride=1):
    # flatten data
    data = input_data.reshape(input_data.shape[0] * input_data.shape[1], input_data.shape[2])
    train_x, train_y = list(), list()
    in_start = 0
    # step over the entire history one STRIDE step at a time
    for _ in range(0, len(data), stride):
        # define the end of the input sequence
        in_end = in_start + n_input_samples
        out_end = in_end + n_output_samples
        # ensure we have enough data for this instance
        if out_end <= len(data):
            train_x.append(data[in_start:in_end, :])
            train_y.append(data[in_end:out_end, 0])
        # move along stride time steps
        in_start += stride
    train_x = np.array(train_x)
    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
    train_x = train_x.reshape(train_x.shape[0], n_timesteps * n_features)
    train_y = np.array(train_y)
    return train_x, train_y

def forecast(model, input_data, n_input_samples):
    # retrieve last n_input observations to predict with
    input_x = input_data[-n_input_samples:, :]
    input_x = input_x.reshape(1, input_x.shape[1]*input_x.shape[0])
    # forecast the next n_output steps
    yhat = model.predict(input_x)
    # we only want the forecast variable
    yhat = yhat[0]
    return yhat

def train_and_make_prediction(input_data, dataset_day, n_output, stride=1):
    dataset = input_data
    n_input_samples = dataset_day * 2 ###TODO: for now two dataset_days are fixed as input!
    n_output_samples = dataset_day * n_output #n_output is in days!

    pipeL = Pipe()
    pipeLine = pipeL.get_pipeline()
    dataset = pipeLine.fit_transform(dataset)

    train_x, train_y = to_supervised(dataset, n_input_samples, n_output_samples, stride)
    model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror')).fit(train_x, train_y)
    yhat_sequence = forecast(model, dataset, n_input_samples)
    yhat_sequence = yhat_sequence.reshape(-1, 1)
    yhat_sequence = pipeLine['Scaler'].inverse_transform(yhat_sequence)
    
    open_hour = str(input_data.index.time.min())
    close_hour = str(input_data.index.time.max())
    times = pd.date_range(start=(input_data.index[len(input_data)-1] + timedelta(hours=1)), periods=24*n_output, freq='H')
    fcst = pd.DataFrame(index=times)
    fcst = fcst.between_time(open_hour, close_hour)
    fcst['yhat'] = yhat_sequence

    return fcst