import pandas as pd
import pytest
from fast_forecaster.processing.data_management import load_dataset


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    file_name = "test.csv"
    return load_dataset(file_name)

