const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/app/api/v1',
    createProxyMiddleware({
      target: 'http://127.0.0.1:8001',
      changeOrigin: true,
    })
  );
};