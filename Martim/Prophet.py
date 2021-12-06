import pandas as pd
from prophet import Prophet
from Main import connect_sqlserver
import matplotlib.pyplot as plt

con,cursor=connect_sqlserver()
query='SELECT startDateHour,tickets from [sisqualFORECASTDATA].[dbo].[MViewTickets] where Rostercode=61'
data = pd.read_sql(query, con)
data=data.rename(columns={'startDateHour':'ds','tickets':'y'})
print(data.head(10))
m = Prophet()
m= Prophet()
m.fit(data)
future = m.make_future_dataframe(periods=365)
print(future.tail(100))
forecast = m.predict(future.tail(100))
fig1 = m.plot(forecast)
plt.show()