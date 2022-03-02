import pandas as pd

from fast_forecaster.config import config

### Serves only for testing purposes
def load_dataset(file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}', header=0, infer_datetime_format=True,parse_dates=['ds'], index_col=['ds'])

    return _data