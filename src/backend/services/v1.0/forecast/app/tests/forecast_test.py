""" Test WFM integration web service.

    project: RH 4.0 FeD / POCI-01-0247-FEDER-039719
	authors: jd
    version: 1.0
	date:    29/11/2022
	
	Test APIs:
		/app/api/v1/models/getforecast/{id}
		/app/api/v1/model/MLPRegressor/getforecast

"""

import requests
import time
precision = '.2f'
sep = '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

url_MLPRegressor_forecast = 'http://127.0.0.1:8000/app/api/v1/model/MLPRegressor/getforecast'
url_forecast = 'http://127.0.0.1:8000/app/api/v1/models/1/forecast'
url_forecast_fail = 'http://127.0.0.1:8000/app/api/v1/model/10/forecast'

token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o'

headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

def MLPRegressor_forecast_test(api_url, request_message):
    return requests.get(api_url, json=request_message)

def forecast_test(api_url, headers, request_message):
    return requests.get(api_url, headers=headers, json=request_message)

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

# prepare imput data for MLPRegressor forecast
client_id = 'AZ20J'
model_imput_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 76, 120, 130, 151, 125, 82, 58, 62, 77, 117, 142, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 72, 102, 139, 149, 137, 69, 48, 64, 82, 111, 148, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 78, 109, 153, 138, 135, 91, 76, 72, 94, 117, 134, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 78, 110, 150, 138, 113, 89, 56, 62, 73, 95, 133, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 65, 82, 140, 127, 130, 83, 54, 58, 99, 142, 142, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 78, 113, 150, 153, 135, 80, 58, 73, 78, 104, 116, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 62, 108, 146, 139, 143, 70, 62, 55, 82, 124, 132, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 47, 71, 113, 143, 136, 123, 77, 60, 57, 80, 122, 168, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 84, 120, 145, 147, 133, 98, 50, 85, 114, 139, 141, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 64, 105, 122, 150, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 76, 119, 152, 169, 133, 104, 65, 60, 108, 150, 116, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 61, 97, 128, 144, 156, 80, 58, 59, 84, 102, 112, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 65, 112, 120, 135, 152, 80, 43, 58, 93, 112, 129, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 76, 112, 129, 174, 137, 91, 71, 59, 92, 118, 125, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 68, 101, 127, 142, 154, 93, 67, 61, 86, 117, 121, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 87, 117, 144, 159, 128, 82, 62, 59, 62, 93, 106, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 65, 99, 113, 178, 127, 95, 74, 81, 99, 137, 133, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 79, 111, 131, 170, 135, 94, 50, 67, 95, 114, 132, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 69, 98, 121, 132, 127, 85, 49, 56, 79, 109, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 60, 105, 130, 150, 138, 74, 58, 67, 91, 133, 130, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 82, 113, 127, 150, 123, 72, 64, 50, 97, 110, 131, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 76, 114, 128, 144, 115, 83, 52, 49, 75, 96, 116, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 85, 121, 130, 130, 126, 98, 51, 80, 98, 122, 116, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 85, 105, 150, 158, 141, 81, 63, 63, 101, 134, 145, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 73, 110, 141, 136, 154, 89, 42, 61, 103, 124, 138, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 86, 127, 143, 151, 112, 78, 52, 80, 112, 147, 147, 59]
forecast_period = 720

model_id = 1
#request_message = {'client_id': client_id, 'model_imput_data': model_imput_data, 'forecast_period': forecast_period}
request_message = {'client_id': client_id, 'model_imput_data': model_imput_data, 'forecast_period': forecast_period}

print('')

# test MLPRegressor_forecast
"""
start_time = time.time()

# send MLPRegressor forecast request
response_message = MLPRegressor_forecast_test(url_MLPRegressor_forecast, request_message)

exec_time = time.time() - start_time

output_response('MLPRegressor Forecast Test', url_MLPRegressor_forecast, response_message, exec_time)
"""

# test forecast
start_time = time.time()

# send forecast request
response_message = forecast_test(url_forecast, headers, request_message)

exec_time = time.time() - start_time

output_response('Forecast Test', url_forecast, response_message, exec_time)

"""
# test forecast fail
start_time = time.time()

# send forecast request
response_message = forecast_test(url_forecast_fail, headers, request_message)

exec_time = time.time() - start_time

output_response('Forecast Test', url_forecast_fail, response_message, exec_time)
"""

print(sep)
