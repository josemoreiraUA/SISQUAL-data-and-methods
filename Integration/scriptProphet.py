import pandas as pd
import numpy as np
import datetime
import pyodbc
from prophet import Prophet
import settings as settings

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

def prophet_forecast(df, start, end, period=60):
    # Create model
    model = Prophet()
    model.fit(df)

    # last day of data
    last_day = pd.to_datetime(df['ds'].iloc[-1]).date()

    # get horizon
    diff = pd.to_datetime(end) - pd.to_datetime(last_day)
    freq = 24 * 60 // period    # 15min -> 96 slots a day; 30min -> 48; 60min -> 24 ; proporcionalidade inversa
    f = model.make_future_dataframe(periods = (diff.days+1)*freq, freq = f'{period}min')

    # forecasting
    fcst = model.predict(f)

    fcst = fcst[['ds','yhat']]
    fcst.set_index('ds', inplace=True)
    fcst = fcst[pd.to_datetime(start):pd.to_datetime(end)+datetime.timedelta(days=1)]

    return fcst

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
fcst = prophet_forecast(df, start, end, period)
pos_df = postprocessing(schedule_df, fcst)

print(pos_df)
