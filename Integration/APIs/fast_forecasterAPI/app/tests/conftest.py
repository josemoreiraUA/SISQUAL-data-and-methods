from typing import Generator

import pandas as pd
import pytest
from fast_forecaster.processing.data_management import load_dataset
from fastapi.testclient import TestClient

from app.main import app

file_name = "test.csv"


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return load_dataset(file_name)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
