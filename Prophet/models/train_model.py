from fbprophet.forecaster import Prophet
import holidays

Prophet(
    growth='linear',
    changepoints=None,
    n_changepoints=25,
    changepoint_range=0.8,
    yearly_seasonality='auto',
    weekly_seasonality='auto',
    daily_seasonality='auto',
    holidays=None,
    seasonality_mode="additive",
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.80,
    uncertainty_samples=1000
    )

def train(model, df):
    model = Prophet()
    model.fit(df)
    return model


def set_holiday(model):
    country_holidays={country:country_code
    , prov:['AN', 'AR', 'AS', 'CB', 'CL'
    , 'CM', 'CN', 'CT', 'EX', 'GA', 'IB'
    , 'MC', 'MD', 'NC', 'PV', 'RI', 'VC']}

    spain_holidays = holidays.Spain(years)
