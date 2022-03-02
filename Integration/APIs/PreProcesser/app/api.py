from datetime import datetime

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter
from loguru import logger

from app import __version__, schemas
from app.config import settings
from app.preprocessing import (
    fill_gaps,
    filter_schedule,
    get_widest_schedule,
    input_inday,
)

api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(name=settings.PROJECT_NAME, api_version=__version__)

    return health.dict()


@api_router.post("/GetHistory", response_model=schemas.History, status_code=200)
async def forecasting_results(rosterCode: str) -> dict:
    """Fetch history for a given Store_ID, Start, and End dates"""
    logger.info(f"Fetching 3 years back of history for store {rosterCode}")

    # TODO: Remove explicit authentication and token from the code.
    auth = "NiCm0ofWFoEwGX9FxWiEn2pD460Zwq3T"
    token = (
        "W1cDGGyKUVnfucuRaAW6JqbZm7LGMxcTG2FuoMtEwnq2Iae6hhWZf3jOZew"
        "j3uHJag9IEK4BzBXsA6pWZSz9LYFKRA7K4feQldsxLtz2opPx6N27v2tliumW8mI4kX8k"
    )
    head = {"Authorization": auth, "AccessToken": token}
    endDate = datetime.date(datetime.now())
    startDate = str(endDate - relativedelta(years=4))
    endDate = str(endDate)
    payload = {"rosterCode": rosterCode, "startDate": startDate, "endDate": endDate}

    # TODO: Remove explicit API URL
    response = requests.get(
        "https://vmi691878/SISQUAL/API/ForecastData/Get",
        headers=head,
        params=payload,
        verify=False,
    ).json()
    store_data_df = pd.DataFrame(response)
    # Verify if dataframe is empty...
    if not store_data_df.empty:
        # if it is then respond differently
        store_data_df = store_data_df.rename(
            columns={"RosterCode": "store", "Tickets": "n_clients", "StartDate": "ds"}
        )
        # TODO: Preprocessing goes here, perhaps create a celery task for this?
        store_data_df = store_data_df[["store", "ds", "n_clients"]]
        store_data_df["ds"] = pd.to_datetime(store_data_df["ds"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        store_data_df["ds"] = pd.to_datetime(
            pd.DatetimeIndex(store_data_df["ds"].astype("string"))
        )
        store_data_df.set_index("ds", inplace=True)
        store_data_df = store_data_df.groupby(["store", pd.Grouper(freq="1H")])[
            "n_clients"
        ].sum()
        store_data_df = store_data_df.reset_index()
        store_data_df.set_index("ds", inplace=True)
        store_data_df = store_data_df[["n_clients"]]
        widest_schedule = get_widest_schedule(store_data_df)
        store_data_df = fill_gaps(store_data_df, widest_schedule)
        store_data_df = filter_schedule(store_data_df, widest_schedule)
        store_data_df = input_inday(store_data_df)
        store_data_df.reset_index(inplace=True)

    # PreProcessed DataFrame response!
    store_history = schemas.History(
        store_id=rosterCode,
        history=store_data_df.to_json(date_format="iso"),
        startDate=startDate,
        endDate=endDate,
        isEmpty=store_data_df.empty,
        nSamplesDay=len(widest_schedule),
    )

    return store_history.dict()
