
import pyodbc 
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
#server = 'deti-sql-aulas.ua.pt' 
#database = 'moreira' 
#username = 'moreira' 
#password = 'jose' 
#cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
#cursor = cnxn.cursor()

import pandas as pd
import pyodbc as odbc

sql_conn = odbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=deti-sql-aulas.ua.pt;DATABASE=moreira;UID=moreira;PWD=jose;')

query = "SELECT * FROM WND_directors"
df = pd.read_sql(query, sql_conn)
print(df.head())
print("abx")
