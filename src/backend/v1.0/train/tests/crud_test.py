""" Test WFM integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
	
	Test APIs:
		/app/api/v1/model/{id}/train

"""

import requests
import time
precision = '.2f'
sep = '--------------------------------------------------------------'

url_get_clients = 'http://127.0.0.1:8001/clients'
url_delete_client = 'http://127.0.0.1:8001/clients/'
url_count_clients = 'http://127.0.0.1:8001/clients/count/'
url_create_client = 'http://127.0.0.1:8001/clients/'
url_get_client = 'http://127.0.0.1:8001/clients/'

token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'
headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

def req(api_url, request_message, headers):
    return requests.post(api_url, json=request_message, headers=headers)

def output_response(info, url, response_message, exec_time):
    # print info
    print(sep)
    print(info)
    print(url)

    # print server reply
    print('')
    print(response_message)
    print('')

    # print forecast
    print(response_message.json())
    print('')

    # print exec timet
    print('exec time: ' + format(exec_time, precision) + ' sec')

print('')

client_id = 'AXZ01'
url_delete_client += client_id

# get clients
start_time = time.time()

response_message = requests.get(url_get_clients, json={}, headers=headers)

exec_time = time.time() - start_time

output_response('List clients Test', url_get_clients, response_message, exec_time)

# count number clients
start_time = time.time()

response_message = requests.get(url_count_clients, json={}, headers=headers)

exec_time = time.time() - start_time

output_response('Count clients Test', url_count_clients, response_message, exec_time)

# delete client
request_message = {'client_id': client_id}

start_time = time.time()

response_message = requests.delete(url_delete_client, json=request_message, headers=headers)

exec_time = time.time() - start_time

output_response('Delete client Test', url_delete_client, response_message, exec_time)

# create client
client_id = 'WF1A7'
client_culture = 'ch'
client_is_active = 0
request_message = {'id': client_id, 'culture': client_culture, 'is_active': client_is_active}

start_time = time.time()

response_message = requests.put(url_create_client, json=request_message, headers=headers)

exec_time = time.time() - start_time

output_response('Create client Test', url_create_client, response_message, exec_time)

# read client
client_id = 'WF1A7'
request_message = {'client_id': client_id}

start_time = time.time()

response_message = requests.get(url_get_client + client_id, json=request_message, headers=headers)

exec_time = time.time() - start_time

output_response('Get client details Test', url_get_client + client_id, response_message, exec_time)

# read non existing client
client_id = 'TF1A7'
request_message = {'client_id': client_id}

start_time = time.time()

response_message = requests.get(url_get_client + client_id, json=request_message, headers=headers)

exec_time = time.time() - start_time

output_response('Get non-existing client details Test', url_get_client + client_id, response_message, exec_time)


url_get_available_models = 'http://127.0.0.1:8001/app/api/v1/models/'
url_client_models = 'http://127.0.0.1:8001/app/api/v1/models/'
url_get_model = 'http://127.0.0.1:8001/models/'
url_create_model = 'http://127.0.0.1:8001/models/'

# available models
start_time = time.time()

response_message = requests.get(url_get_available_models, json={}, headers=headers)

exec_time = time.time() - start_time

output_response('List available models Test', url_get_available_models, response_message, exec_time)

# create model

client_id = 'WF1A7'
client_culture = 'ch'
client_is_active = 0

request_message = {'type': '1' \
                 , 'storage_name': 'mlp' \
				 , 'metrics': 'rms:1.5' \
				 , 'forecast_period': '720' \
				 , 'train_params': 'k:9;it:10' \
				 }

start_time = time.time()

response_message = requests.put(url_create_model + 'WF1A7', json=request_message, headers=headers)

exec_time = time.time() - start_time

output_response('Create model Test', url_create_model + 'WF1A7', response_message, exec_time)


# client models
start_time = time.time()

response_message = requests.get(url_client_models + 'WF1A7', json={}, headers=headers)

exec_time = time.time() - start_time

output_response('List client models Test', url_client_models + 'WF1A7', response_message, exec_time)

# client models
start_time = time.time()

response_message = requests.get(url_client_models + 'Ax01', json={}, headers=headers)

exec_time = time.time() - start_time

output_response('List client models Test', url_client_models + 'Ax01', response_message, exec_time)

# get model
start_time = time.time()

response_message = requests.get(url_get_model + '1', json={}, headers=headers)

exec_time = time.time() - start_time

output_response('Get model Test', url_get_model + '1', response_message, exec_time)

# get model
start_time = time.time()

response_message = requests.get(url_get_model + '100', json={}, headers=headers)

exec_time = time.time() - start_time

output_response('Get model Test', url_get_model + '100', response_message, exec_time)

url_get_task = 'http://127.0.0.1:8001/tasks/'
url_get_task_state = 'http://127.0.0.1:8001/tasks/1/state/'
url_create_task = 'http://127.0.0.1:8001/tasks/'

# create task
request_message = {'client_pkey': 1 \
                 , 'model_type': '1' \
				 , 'state': 'pending'
				 }

start_time = time.time()

response_message = requests.put(url_create_task, json=request_message, headers=headers)

exec_time = time.time() - start_time

output_response('Create task Test', url_create_task, response_message, exec_time)


# get task
start_time = time.time()

response_message = requests.get(url_get_task + '1', json={}, headers=headers)

exec_time = time.time() - start_time

output_response('Get task Test', url_get_task + '1', response_message, exec_time)

# get task state
start_time = time.time()

response_message = requests.get(url_get_task_state, json={}, headers=headers)

exec_time = time.time() - start_time

output_response('Get task state Test', url_get_task_state, response_message, exec_time)

print(sep)
