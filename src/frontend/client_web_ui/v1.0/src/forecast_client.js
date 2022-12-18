import forecast_client from './forecast_config';

const axios = require('axios');

class forecastAPIClient {
  constructor(overrides) {
    this.config = {
      ...forecast_client,
      ...overrides,
    };
	
    this.apiClient = this.getApiClient(this.config);
  }
  
  /* ----- Client Configuration ----- */

  /* Create Axios client instance pointing at the REST api backend */
  getApiClient(forecast_client) {
    const initialConfig = {
      baseURL: `${forecast_client.apiBasePath}/api/v1/app`,
    };
    const client = axios.create(initialConfig);
    client.interceptors.request.use(localStorageTokenInterceptor);
    return client;
  }

  forecast(modelId, json_payload){
    
    return this.apiClient.post(`/models/${modelId}/forecast`, json_payload).then(({data}) => {
      return data;
    }).catch((error) => console.log(error));
  } 
}

// every request is intercepted and has auth header injected.
function localStorageTokenInterceptor(config) {
  
  const headers = {};
  const tokenString = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOjE2NjY2MzMxMjYsImV4cCI6MTg1NjAyMTkyNiwiaXNzIjoiU0lTUVVBTFdGTSM3MDAxIiwiYXVkIjoiRm9yZWNhc3RNYW5hZ2VyIn0.qii3Q-Ocp1YDs9ASTiZNEzFNiDu7Ia3ZxOCUNRQgJ_o';
  headers['Authorization'] = `Bearer ${tokenString}`;
  
  //const tokenString = localStorage.getItem('token');

  config['headers'] = headers;
  return config;
}

export default forecastAPIClient;