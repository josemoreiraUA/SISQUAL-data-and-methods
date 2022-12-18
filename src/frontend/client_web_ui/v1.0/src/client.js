import config from './config';

const axios = require('axios');

class FastAPIClient {
  constructor(overrides) {
    this.config = {
      ...config,
      ...overrides,
    };
	
    this.apiClient = this.getApiClient(this.config);
  }

  /* ----- Client Configuration ----- */

  /* Create Axios client instance pointing at the REST api backend */
  getApiClient(config) {
    const initialConfig = {
      baseURL: `${config.apiBasePath}/app/api/v1`,
    };
    const client = axios.create(initialConfig);
    client.interceptors.request.use(localStorageTokenInterceptor);
    return client;
  }

  get_list_available_forecast_models() {
    return this.apiClient.get(`/models/`).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  }

  get_list_clients() {
    return this.apiClient.get(`/clients/`).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  }

  get_client_parameters(clientId){
    return this.apiClient.get(`/clients/${clientId}`).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  }

  get_client_trained_models(clientId){
    return this.apiClient.get(`/client/${clientId}/models`).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  }
  
  get_client_tasks(clientId){
    return this.apiClient.get(`/client/${clientId}/tasks`).then(({data}) => {
      return data;
    }).catch((err) => {
		alert(err);
		console.log(err);
	});
  }  
  
  update_client_parameters(clientId, json_payload){
    return this.apiClient.post(`/clients/${clientId}`,json_payload).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  }

  create_client(json_payload){
    return this.apiClient.put(`/clients/`,json_payload).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  }  
  
  train_model(modelType, json_payload){
    return this.apiClient.post(`/models/${modelType}/train/`,json_payload).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  } 
}

// every request is intercepted and has auth header injected.
function localStorageTokenInterceptor(config) {
  
  const headers = {};
  const tokenString = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o';
  headers['Authorization'] = `Bearer ${tokenString}`;

  config['headers'] = headers;
  return config;
}

export default FastAPIClient;
