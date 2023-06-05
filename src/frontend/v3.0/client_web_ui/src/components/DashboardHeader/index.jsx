import React from 'react';
import {Link} from 'react-router-dom';

function DashboardHeader() {
  return (
    <nav className="flex items-center shadow-lg justify-between flex-wrap bg-gray-50 p-4">
        <div className="flex items-center flex-shrink-0 mr-2">
			<span 
				className="font-bold text-xl tracking-tight"
				style={{ color: 'red' }}>
				F
			</span>
            <span className="font-semibold text-xl tracking-tight">
				eD 4.0 Prototype UI
			</span>
        </div>
        <div className="">
            <Link to="/"
                className="font-semibold block mt-2 lg:inline-block lg:mt-0 hover:text-blue-600 mx-1 transition duration-150 ease-in-out">
                Home
            </Link>
        </div>
    </nav>
  );
}

export default DashboardHeader;
