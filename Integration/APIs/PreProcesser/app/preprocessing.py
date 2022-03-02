from datetime import datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd


def get_widest_schedule(df) -> list:
    check_dict = dict()
    # iterate by days
    for idx, day in df.groupby(df.index.date):
        day_sched = list()
        # get list(keys-schedules) for the dict
        for i in day.index.time:
            hour = i.strftime("%H:%M:%S")
            day_sched.append(hour)
        day_sched = str(day_sched)
        if day_sched in check_dict:
            check_dict[day_sched] += 1
        else:
            check_dict.update({day_sched: 1})

    # the widestst working schedule
    return eval(max(check_dict.keys(), key=len))


def fill_gaps(df, widest_schedule=None, hour_in=None, hour_out=None) -> pd.DataFrame:
    counter = 0  # counter of imputations to perform
    # create custom schedule with hour_in and hour_out
    if hour_in is not None and hour_out is not None:
        hour_in = datetime.strptime(hour_in, "%H:%M:%S")
        hour_out = datetime.strptime(hour_out, "%H:%M:%S")
        widest_schedule = list()
        while hour_in <= hour_out:
            widest_schedule.append(
                str(datetime.strptime(str(hour_in.time()), "%H:%M:%S").time())
            )
            hour_in += timedelta(minutes=60)

    # fill the gaps with NaN
    for i in widest_schedule:
        i = datetime.strptime(i, "%H:%M:%S").time()
        for j, day in df.groupby(df.index.date):
            if i not in list(day.index.time):
                df.loc[pd.to_datetime(str(j) + " " + str(i))] = [np.nan]
                counter += 1

    # add column "imputed"
    df["imputed"] = np.where(((pd.isnull(df["n_clients"]))), "yes", "no")
    df = df.sort_index()

    # Dataframe with NaNs to be imputed and excess datetimes
    return df


# Remove non-regular time rows
def filter_schedule(df, widest_schedule) -> pd.DataFrame:
    # convert schedule to datetime pd.Series
    pd_schedule = pd.to_datetime(pd.Series(widest_schedule))
    # create a sequence of indexed halfhours
    pd_schedule = (
        pd_schedule.dt.strftime("%H").astype("int64") * 2
        + pd_schedule.dt.strftime("%M").astype("int64") // 30
    )

    # check if the schedule is continuous
    if max(pd_schedule) - min(pd_schedule) + 1 == len(pd_schedule):
        # no breaks
        df = df.between_time(widest_schedule[0], widest_schedule[-1])
    else:
        # there is a break
        df_schedule = pd.DataFrame({"time": widest_schedule, "hh": pd_schedule})
        # find break
        for i in df_schedule.index:
            if (df_schedule.at[i + 1, "hh"] - df_schedule.at[i, "hh"]) != 1:
                break
        close_morning = df_schedule.at[i, "time"]
        open_afternoon = df_schedule.at[i + 1, "time"]
        # make 2 splits
        df1 = df.between_time(widest_schedule[0], close_morning)
        df2 = df.between_time(open_afternoon, widest_schedule[-1])
        # join
        df = pd.concat([df1, df2]).sort_index()

    return df


# Imput missing data day-by-day
def input_inday(df, method="linear") -> pd.DataFrame:
    # get list of days
    for dt in np.unique(df.index.date):
        # input each day individualy
        df.loc[str(dt)] = df.loc[str(dt)].interpolate(
            method=method, limit_direction="both"
        )

    return df
