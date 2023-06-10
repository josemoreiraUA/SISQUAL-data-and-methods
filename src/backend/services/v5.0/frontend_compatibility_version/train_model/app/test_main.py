from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.dependencies.db import get_db
from app.main import train_app
from app.core.config import settings

# create databse for testing >>
#SQLALCHEMY_DATABASE_URL = "sqlite:///.../test.db"
SQLALCHEMY_DATABASE_URL = "sqlite:///C:/temp/backend/services/test.db"
#SQLALCHEMY_DATABASE_URL = "sqlite:///C:/temp/test.db"


engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

train_app.dependency_overrides[get_db] = override_get_db
# <<

client = TestClient(train_app)
request_prefix = '/api/v1'

model_id = 0
task_id = 0
client_pkey = 0

# ---------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------

def test_create_client():
    client_id = 'AR21TY2023'
    culture = 'en-EN'
    is_active = False

    response = client.put(f'{request_prefix}/clients', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'id': client_id, 'culture': culture, 'is_active': is_active})
    
    assert response.status_code == 200
    assert response.json()['detail'] == '1', f'Error creating client!'

def test_get_client_details():
    client_id = 'AR21TY2023'
    culture = 'en-EN'
    is_active = False

    response = client.get(f'{request_prefix}/clients/{client_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    global client_pkey

    client_pkey = response.json()['client_pkey']

    assert response.status_code == 200
    assert response.json()['id'] == f'{client_id}'
    assert response.json()['culture'] == f'{culture}'
    assert response.json()['is_active'] == is_active

def test_update_client_parameters():
    client_id = 'AR21TY2023'
    culture = 'pt-ES'
    is_active = True

    response = client.post(f'{request_prefix}/clients/{client_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'culture': culture, 'is_active': is_active})
    
    assert response.status_code == 200
    assert response.json()['detail'] == '1', f'API should allow changing client parameters!'

    response = client.get(f'{request_prefix}/clients/{client_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 200
    assert response.json()['id'] == f'{client_id}'
    assert response.json()['culture'] == f'{culture}'
    assert response.json()['is_active'] == is_active

def test_get_list_knwon_clients():
    response = client.get(f'{request_prefix}/clients', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 200
    assert len(response.json()['clients']) == 1, 'Only one client is registered.'

def test_create_client_with_already_existing_id():
    client_id = 'AR21TY2023'
    culture = 'en-EN'
    is_active = False

    response = client.put(f'{request_prefix}/clients', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'id': client_id, 'culture': culture, 'is_active': is_active})
    
    assert response.status_code == 409
    assert response.json()['detail'] == f'Client {client_id} already exists!', f'{client_id} already exists. It should not be possible to create two clients with the same id!'

def test_get_non_existing_client_details():
    client_id = '00000023'
    response = client.get(f'{request_prefix}/clients/{client_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 404
    assert response.json()['detail'] == f'Client {client_id} not found!', f'{client_id} not supposed to exist!'

def test_update_non_existing_client_parameters():
    client_id = '00000023'
    culture = 'en-EN'
    is_active = False

    response = client.post(f'{request_prefix}/clients/{client_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'culture': culture, 'is_active': is_active})
    
    assert response.status_code == 404
    assert response.json()['detail'] == f'Client {client_id} not found!', f'{client_id} not supposed to exist!'

def test_get_non_existing_client_models():
    client_id = '00000023'
    response = client.get(f'{request_prefix}/clients/{client_id}/models', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 404
    assert response.json()['detail'] == f'Client {client_id} not found!', f'{client_id} not supposed to exist!'

def test_get_non_existing_client_tasks():
    client_id = '00000023'
    response = client.get(f'{request_prefix}/clients/{client_id}/tasks', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 404
    assert response.json()['detail'] == f'Client {client_id} not found!', f'{client_id} not supposed to exist!'

def test_get_list_available_models():
    response = client.get(f'{request_prefix}/models', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 200
    assert len(response.json()['models']) == 5
    assert response.json() == {'models':[{'model_type':'MLPRegressor','model_name':'MLPRegressor'},{'model_type':'HistGradientBoostingRegressor','model_name':'HistGradientBoostingRegressor'},{'model_type':'Prophet','model_name':'Prophet'},{'model_type':'RandomForestRegressor','model_name':'RandomForestRegressor'}, {'model_name': 'MLPRegressor2', 'model_type': 'MLPRegressor2'}]}

def test_get_non_existing_task_state():
    task_id = 0
    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 404
    assert response.json()['detail'] == f'Task {task_id} not found!'

def test_get_non_existing_task_details():
    task_id = 0
    response = client.get(f'{request_prefix}/tasks/{task_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 404
    assert response.json()['detail'] == f'Task {task_id} not found!'

def test_get_non_existing_model_details():
    model_id = 0
    response = client.get(f'{request_prefix}/models/{model_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 404
    assert response.json()['detail'] == f'Model {model_id} not found!'

def test_train_model_non_existing_client():
    client_id = '00000023'
    model_name = 'Sales'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    n_lags = 720
    model_type = 'MLPRegressor'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 404
    assert response.json()['detail'] == f'Client {client_id} not found!', f'{client_id} not supposed to exist!'

def test_train_model_non_availanle_model():
    client_id = 'AR21TY2023'
    model_name = 'Sales'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    n_lags = 720
    model_type = 'MLPRegressor3'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 404
    assert response.json()['detail'] == f'Model type {model_type} not available!', f'{model_type} not supposed to be available!'

def test_train_MLPRegressor_model():
    client_id = 'AR21TY2023'
    model_name = 'Store A'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    n_lags = 720
    model_type = 'MLPRegressor'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    global task_id
    task_id = response.json()['task_id']

def test_get_existing_model_details():
    model_id = 1
    model_name = 'Store A'
    response = client.get(f'{request_prefix}/models/{model_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['id'] == model_id
    assert response.json()['type'] == 'MLPRegressor'
    assert response.json()['model_name'] == model_name

def test_get_existing_task_details():
    response = client.get(f'{request_prefix}/tasks/{task_id}', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['id'] == task_id
    assert response.json()['client_pkey'] == client_pkey
    assert response.json()['model_type'] == 'MLPRegressor'
    assert response.json()['state'] == 'Finished'

def test_get_existing_task_state():
    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200

    if response.json()['state'] == 'Finished':
        assert len(response.json()['html_report']) > 0, 'When finished, a task should send a html report.'
    elif response.json()['state'] == 'Error':
        assert len(response.json()['error_report']) > 0, 'When an error occurs, a task should send an error report.'

    assert response.json()['state'] in ['Created', 'Pending', 'Executing', 'Error', 'Finished'], f'{response.json()["state"]} is an unknown task state.'
    assert response.json()['state'] == 'Finished'

def test_train_HistGradientBoostingRegressor_model():
    client_id = 'AR21TY2023'
    model_name = 'Sales Store B'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    n_lags = 720
    model_type = 'HistGradientBoostingRegressor'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

def test_train_Prophet_model():
    client_id = 'AR21TY2023'
    model_name = 'Sales Store C'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    n_lags = 720
    model_type = 'Prophet'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

def test_get_client_models():
    client_id = 'AR21TY2023'
    response = client.get(f'{request_prefix}/clients/{client_id}/models', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 200
    assert len(response.json()['models']) == 3, f'{client_id} is expected to have 3 models!'

def test_get_client_tasks():
    client_id = 'AR21TY2023'
    response = client.get(f'{request_prefix}/clients/{client_id}/tasks', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})
    
    assert response.status_code == 200
    assert len(response.json()['tasks']) == 3, f'{client_id} is expected to have 3 tasks!'

def test_train_MLPRegressor_model_with_min_data_size():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 720
    n_lags = 1440
    model_type = 'MLPRegressor'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    print(data_resample)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (n_lags + forecast_period) + 1

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Finished'
    assert len(response.json()['html_report']) > 0, 'When finished, a task should send a html report.'

    print(response.json())

def test_train_MLPRegressor_model_with_min_data_size_forecast_period_bigger_than_n_lags():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 1440
    n_lags = 720
    model_type = 'MLPRegressor'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    print(data_resample)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (n_lags + forecast_period) + 1

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Finished'
    assert len(response.json()['html_report']) > 0, 'When finished, a task should send a html report.'

    print(response.json())

def test_train_MLPRegressor_model_with_less_than_min_data_size():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 720
    n_lags = 1440
    model_type = 'MLPRegressor'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    print(data_resample)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (n_lags + forecast_period) - 1

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Error'
    assert len(response.json()['error_report']) > 0, 'When finished, a task should send an error report.'
    assert response.json()['error_report'] == '<div><h3>Task 6 Error: Training data size (2159 hours) too small (must be at least 2161 hours)!</h3></div>'

    print(response.json())

def test_train_MLPRegressor_model_with_less_than_min_data_size_equal_zero():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 720
    n_lags = 1440
    model_type = 'MLPRegressor'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (n_lags + forecast_period)

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'n_lags': n_lags})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Error'
    assert len(response.json()['error_report']) > 0, 'When finished, a task should send an error report.'
    assert response.json()['error_report'] == '<div><h3>Task 7 Error: Training data size (2160 hours) too small (must be at least 2161 hours)!</h3></div>'

    print(response.json())

def test_train_HistGradientBoostingRegressor_model_with_min_data_size():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 720
    model_type = 'HistGradientBoostingRegressor'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    print(data_resample)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (forecast_period) + 1

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Finished'
    assert len(response.json()['html_report']) > 0, 'When finished, a task should send a html report.'

    print(response.json())

def test_train_HistGradientBoostingRegressor_model_with_less_then_min_data_size():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 720
    model_type = 'HistGradientBoostingRegressor'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    print(data_resample)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (forecast_period)

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Error'
    assert len(response.json()['error_report']) > 0, 'When finished, a task should send an error report.'
    assert response.json()['error_report'] == f'<div><h3>Task 9 Error: Training data size ({N} hours) too small (must be at least {forecast_period + 1} hours)!</h3></div>'

    print(response.json())

def test_train_Prophet_model_with_less_then_min_data_size():
    client_id = 'AR21TY2023'
    model_name = 'Store F'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')
    forecast_period = 720
    model_type = 'Prophet'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    data_resample = pd.DataFrame(data.resample('H', on = 'ds').y.sum())
    data_resample.reset_index(inplace=True)

    print(data_resample)

    ds = data_resample['ds'].astype('string').tolist()
    y = data_resample['y'].tolist()

    N = (forecast_period) + 2

    ds = ds[0:N]
    y = y[0:N]

    assert len(ds) == len(y)
    assert len(ds) == N

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Finished'
    assert len(response.json()['html_report']) > 0, 'When finished, a task should send a html report.'

    print(response.json())

def test_train_Prophet_model_unexpected_input_keys():
    client_id = 'AR21TY2023'
    model_name = 'Sales Store C'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y': y, 'x': [0, 2, 3]}
    forecast_period = 720
    model_type = 'Prophet'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 400
    assert response.json() == {'detail': 'Unexpected input data structure!'}

def test_train_Prophet_model_unexpected_input_key():
    client_id = 'AR21TY2023'
    model_name = 'Sales Store C'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()

    input_data = {'ds' : ds, 'y1': y}
    forecast_period = 720
    model_type = 'Prophet'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 400
    assert response.json() == {'detail': 'Unexpected input data structure!'}

def test_train_Prophet_model_unexpected_y_input_data():
    client_id = 'AR21TY2023'
    model_name = 'Sales Store C'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()
    y[0] = 'a'

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    model_type = 'Prophet'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    print(response.json())

    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Error'
    assert response.json()['error_report'] == '<div><h3>Task 11 Error: can only concatenate str (not "float") to str!</h3></div>'

def test_train_Prophet_model_unexpected_ds_input_data():
    client_id = 'AR21TY2023'
    model_name = 'Sales Store C'
    
    import pandas as pd

    data = pd.read_csv('train_data.csv')

    ds = data['ds'].astype('string').tolist()
    y = data['y'].tolist()
    ds[0] = 'a'

    input_data = {'ds' : ds, 'y': y}
    forecast_period = 720
    model_type = 'Prophet'

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    print(response.json())

    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] == 'Error'
    assert response.json()['error_report'] == '<div><h3>Task 12 Error: Given date string a not likely a datetime present at position 0!</h3></div>'

# ---------------------------------------------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------------------------------------------

def test_train_RandomForestRegressor_model():
    client_id = 'AR21TY2023'
    model_name = 'Store F Team Sizing'
    
    import pandas as pd

    data = pd.read_csv('test_store_sizing_data.csv')

    forecast_period = 720
    model_type = 'RandomForestRegressor'

    data['StartDate'] = pd.to_datetime(data['StartDate'])
    data.shape

    data.reset_index(inplace=True)
    data.rename(columns={'StartDate': 'ds', 'Tickets': 'x', 'Checkouts': 'y'}, inplace=True)    

    ds = data['ds'].astype('string').tolist()
    x = data['x'].tolist()
    y = data['y'].tolist()

    #print(ds)
    #print(x)
    #print(y)

    N = forecast_period

    ds = ds[:-forecast_period]
    x = x[:-forecast_period]
    y = y[:-forecast_period]

    assert len(ds) == len(x)
    assert len(ds) == len(y)

    input_data = {'ds' : ds, 'x': x, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] != 'Error'

    print(response.json())

def test_train_RandomForestRegressor_model_with_less_imput_data_parameters():
    client_id = 'AR21TY2023'
    model_name = 'Store F Team Sizing'
    
    import pandas as pd

    data = pd.read_csv('test_store_sizing_data.csv')

    forecast_period = 720
    model_type = 'RandomForestRegressor'

    data['StartDate'] = pd.to_datetime(data['StartDate'])
    data.shape

    data.reset_index(inplace=True)
    data.rename(columns={'StartDate': 'ds', 'Tickets': 'x', 'Checkouts': 'y'}, inplace=True)    

    ds = data['ds'].astype('string').tolist()
    x = data['x'].tolist()
    y = data['y'].tolist()

    N = forecast_period

    ds = ds[0:-forecast_period]
    x = x[0:-forecast_period]
    y = y[0:-forecast_period]

    assert len(ds) == len(x)
    assert len(ds) == len(y)

    input_data = {'ds' : ds, 'y': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 400
    assert response.json()['detail'] == 'Unexpected input data structure!'

    print(response.json())

def test_train_RandomForestRegressor_model_with_bad_name_imput_data_parameters():
    client_id = 'AR21TY2023'
    model_name = 'Store F Team Sizing'
    
    import pandas as pd

    data = pd.read_csv('test_store_sizing_data.csv')

    forecast_period = 720
    model_type = 'RandomForestRegressor'

    data['StartDate'] = pd.to_datetime(data['StartDate'])
    data.shape

    data.reset_index(inplace=True)
    data.rename(columns={'StartDate': 'ds', 'Tickets': 'x', 'Checkouts': 'y'}, inplace=True)    

    ds = data['ds'].astype('string').tolist()
    x = data['x'].tolist()
    y = data['y'].tolist()

    N = forecast_period

    ds = ds[0:-forecast_period]
    x = x[0:-forecast_period]
    y = y[0:-forecast_period]

    assert len(ds) == len(x)
    assert len(ds) == len(y)

    input_data = {'ds' : ds, 'x': x, 'a': y}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period})
    
    assert response.status_code == 400
    assert response.json()['detail'] == 'Unexpected input data structure!'

    print(response.json())

def test_train_MLPRegressor_model_tickets_and_sizing():
    client_id = 'AR21TY2023'
    model_name = 'Store F Tickets and Team Sizing'
    
    import pandas as pd

    data = pd.read_csv('test_store_tickets_and_sizing_data.csv')

    forecast_period = 720
    model_type = 'MLPRegressor2'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    ds = data['ds'].astype('string').tolist()
    y1 = data['y1'].tolist()
    y2 = data['y2'].tolist()

    N = forecast_period

    """
    ds10 = ds[:-forecast_period]
    y110 = y1[:-forecast_period]
    y220 = y2[:-forecast_period]

    print(ds10)
    print(y110)
    print(y220)
    print('')
    """

    """
    ds11 = ds[-forecast_period*2:-forecast_period]
    y111 = y1[-forecast_period*2:-forecast_period]
    y221 = y2[-forecast_period*2:-forecast_period]

    print(ds11)
    print(y111)
    print(y221)
    print('')

    ds12 = ds[-forecast_period:]
    y112 = y1[-forecast_period:]
    y222 = y2[-forecast_period:]

    print(ds12)
    print(y112)
    print(y222)
    print('')
    """

    ds = ds[:-forecast_period]
    y1 = y1[:-forecast_period]
    y2 = y2[:-forecast_period]

    assert len(ds) == len(y1)
    assert len(ds) == len(y2)

    input_data = {'ds' : ds, 'y1': y1, 'y2': y2}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'customer_flow_and_team_size': True})
    
    assert response.status_code == 202
    assert response.json()['detail'] == '1'
    assert response.json()['task_id'] > 0

    print(response.json())

    task_id = response.json()['task_id']

    response = client.get(f'{request_prefix}/tasks/{task_id}/state', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN})

    assert response.status_code == 200
    assert response.json()['state'] != 'Error'

    print(response.json())

def test_train_MLPRegressor_model_tickets_and_sizing_with_less_number_parameters():
    client_id = 'AR21TY2023'
    model_name = 'Store F Tickets and Team Sizing'
    
    import pandas as pd

    data = pd.read_csv('test_store_tickets_and_sizing_data.csv')

    forecast_period = 720
    model_type = 'MLPRegressor2'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    ds = data['ds'].astype('string').tolist()
    y1 = data['y1'].tolist()
    y2 = data['y2'].tolist()

    N = forecast_period

    ds = ds[:-forecast_period]
    y1 = y1[:-forecast_period]
    y2 = y2[:-forecast_period]

    assert len(ds) == len(y1)
    assert len(ds) == len(y2)

    input_data = {'ds' : ds, 'y2': y2}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'customer_flow_and_team_size': True})
    
    assert response.status_code == 400
    assert response.json()['detail'] == 'Unexpected input data structure!'

    print(response.json())

def test_train_MLPRegressor_model_tickets_and_sizing_with_bad_name_parameters():
    client_id = 'AR21TY2023'
    model_name = 'Store F Tickets and Team Sizing'
    
    import pandas as pd

    data = pd.read_csv('test_store_tickets_and_sizing_data.csv')

    forecast_period = 720
    model_type = 'MLPRegressor2'

    data['ds'] = pd.to_datetime(data['ds'])
    data.shape

    ds = data['ds'].astype('string').tolist()
    y1 = data['y1'].tolist()
    y2 = data['y2'].tolist()

    N = forecast_period

    ds = ds[:-forecast_period]
    y1 = y1[:-forecast_period]
    y2 = y2[:-forecast_period]

    assert len(ds) == len(y1)
    assert len(ds) == len(y2)

    input_data = {'ds' : ds, 'x': y1, 'y2': y2}

    response = client.post(f'{request_prefix}/models/{model_type}/train', headers={'Authorization': 'Bearer ' + settings.JWT_TOKEN}, json={'client_id': client_id, 'model_name': model_name, 'input_data': input_data, 'forecast_period': forecast_period, 'customer_flow_and_team_size': True})
    
    assert response.status_code == 400
    assert response.json()['detail'] == 'Unexpected input data structure!'

    print(response.json())