import numpy as np
import datetime
import pandas as pd
import os

data = pd.read_csv('timeseries.csv')
#data.to_csv('ori.csv')

new_val = data.pivot_table(index = ['ts'], aggfunc ='size')
#print(new_val)
new_val.to_csv('dup_count.csv')

#data.sort_values(by=['ts'], inplace = True)
#data.to_csv('sorted.csv')

ts = data['ts']
print('initial data size: ' + str(ts.size))

new_val = data.duplicated(subset=['ts'], keep='first')

new_output = data.loc[new_val==True].copy()
new_output.sort_values(by=['ts'], inplace = True)
new_output.to_csv('dup_t.csv')

#print('duplicates: ' + str(new_output.size))

new_output = data.loc[new_val==False].copy()
new_output.sort_values(by=['ts'], inplace = True)
new_output.to_csv('dup_f.csv')

#print('unique: ' + str(new_output.size))


#new_output = data[data.duplicated('ts')]
#print("Duplicated values", new_output)

#new_output.to_csv('dup.csv')

"""
print('-----------------------------------------------')

#duplicate_rows = ts.duplicated(keep=False)
duplicate_rows = ts.duplicated(keep='first')
data['duplicate_rows'] = duplicate_rows

#for row in data.itertuples():
#	print(row)

unique = 0
dup = 0
for elem in duplicate_rows:
	if elem:
		unique += 1
	else:
		dup += 1

print('unique: ', unique, 'dup', dup)
"""
#print(duplicate_rows)

#print('num duplicated rows: ' + str(duplicate_rows.size))