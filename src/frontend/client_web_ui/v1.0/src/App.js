import React from 'react';
import './App.css';
import {BrowserRouter, Routes, Route} from 'react-router-dom';
import Home from './pages/home';
import Clients from './pages/clients';
import ClientDashboard from './pages/client-dashboard';
import ErrorPage from './pages/error-page';

const App = () => {
  return (
    <div className="App bg-black">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route exact path="/clients" element={<Clients />} />
          <Route exact path="/client-dashboard" element={<ClientDashboard />} / >
          <Route exact={true} path="*" element={<ErrorPage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
};

export default App;
