import React from 'react';
import { Link } from "react-router-dom";
const ErrorPage = () => {
     return (
          <div className="flex items-center justify-center w-screen h-screen bg-slate-900"
          >
				<div className="px-40 py-20 bg-white rounded-md shadow-xl">
                    <div className="flex flex-col items-center">
                        <h1 className="font-bold text-blue-800 text-9xl">404</h1>
                        <h6 className="mb-2 text-2xl font-bold text-center text-gray-800 md:text-3xl">
                            Page not found.
                        </h6>
                        <p className="mb-8 text-center text-gray-500 md:text-lg">
                            The page you’re looking for doesn’t exist.
                        </p>
                        <Link
                            to="/"
                            className="px-6 py-2 text-sm font-semibold text-blue-800 bg-slate-100"
                        >
							Home
						</Link >
                    </div>
				</div>
          </div>
     )
}

export default ErrorPage;
