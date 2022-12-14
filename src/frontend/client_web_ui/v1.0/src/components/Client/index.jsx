import React from "react";
import {Link} from 'react-router-dom';

const Client = ({client}) => {
	return (
		client && (
			<>
				<div
					onClick={(e) => {e.stopPropagation()}} 
					className="flex flex-wrap items-end justify-between w-full bg-white mb-1">
						<div className="w-full shadow-lg">
							<div 
								className="relative hover:bg-slate-200 active:bg-slate-500 flex flex-col h-full p-1"
								style={{ hover: '#161730' }}
							>
								<Link 
									to='/client-dashboard' 
									state={{id:`${client?.id}`}}
									className="px-4 text-left font-bold block mt-2 lg:inline-block lg:mt-0"
									>
										{client?.id}
								</Link>
							</div>
						</div>
				</div>
			</>
		)
	);
};

export default Client;
