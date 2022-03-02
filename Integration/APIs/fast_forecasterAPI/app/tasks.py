import pandas as pd
from fast_forecaster.train_predict import train_and_make_prediction
from fastapi.responses import JSONResponse
from app.celery_app import app_celery


@app_celery.task(name="train_and_predict")
def predict_task(input_df_json: str, dataset_day: int, n_output: int) -> JSONResponse:
    input_df = pd.read_json(input_df_json, convert_dates=False)
    input_df['ds'] = pd.to_datetime(input_df['ds'])
    input_df.set_index('ds', inplace=True)
    # TODO: preprocess ds column
    results_df = train_and_make_prediction(input_df, dataset_day, n_output)
    return results_df.to_json(date_format='iso')
