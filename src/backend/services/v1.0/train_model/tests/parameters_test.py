""" 
	Test WFM integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
	
	Test APIs:
		/app/api/v1/parameters/{id}
"""

import requests
import time
precision = '.2f'

url_parameters = 'http://127.0.0.1:8000/app/api/v1/parameters'
sep = '-------------------------------------------------------------------------------------------------------'

def output_response(info, url, type, response_message, exec_time):
    # print info
    print(sep)
    print(info)
    print(url + ' (' + type + ')')

    # print server reply
    print('')
    print(response_message)
    #print(response_message.headers)
    print('')

    # print forecast
    print(response_message.json())
    print('')

    # print exec timet
    print('exec time: ' + format(exec_time, precision) + ' sec')

print('')

# prepare imput data
client_id = 1
request_message = {'culture': 'pt', 'forecast_period': 720}

token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'
token1 = 'EyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'

headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
headers1 = {'Content-Type': 'application/json'}
headers2 = {'Authorization': 'Bearer ' + token1, 'Content-Type': 'application/json'}

# insert
start_time = time.time()

url_parameters = url_parameters + '/' + str(client_id)

response_message = requests.post(url_parameters, headers=headers, json=request_message)

exec_time = time.time() - start_time

output_response('Insert', url_parameters, 'post', response_message, exec_time)

# insert 2
start_time = time.time()

response_message = requests.post(url_parameters, headers=headers1, json=request_message)

exec_time = time.time() - start_time

output_response('Insert', url_parameters, 'post', response_message, exec_time)

# insert 3
start_time = time.time()

response_message = requests.post(url_parameters, headers=headers2, json=request_message)

exec_time = time.time() - start_time

output_response('Insert', url_parameters, 'post', response_message, exec_time)

# get
start_time = time.time()

response_message = requests.get(url_parameters, headers=headers)

exec_time = time.time() - start_time

output_response('Get', url_parameters, 'get', response_message, exec_time)

# update
start_time = time.time()

response_message = requests.put(url_parameters, headers=headers, json=request_message)

exec_time = time.time() - start_time

output_response('Update', url_parameters, 'put', response_message, exec_time)

# delete
start_time = time.time()

response_message = requests.delete(url_parameters, headers=headers)

exec_time = time.time() - start_time

output_response('Delete', url_parameters, 'delete', response_message, exec_time)

print(sep)