import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from fast_forecaster.train_predict import train_and_make_prediction
from fast_forecaster.processing.data_management import load_dataset

TARGET = 'n_clients'
PREDICTION = 'yhat'
def test_make_single_prediction(test_data: pd.DataFrame) -> None:
    # Given
    dataset_day=13
    n_output=10 # n_output is in days! A ~10 days forecast!
    test_data = test_data[:-(dataset_day*n_output)]
    print(test_data.tail())
    # When
    subject = train_and_make_prediction(test_data, dataset_day, n_output)

    # Then
    print(subject.head())
    print(r2_score(test_data[TARGET].values[-(dataset_day*n_output):], subject[PREDICTION].values))
    assert subject is not None
    assert isinstance(subject[PREDICTION].values.flat[0], np.floating)
