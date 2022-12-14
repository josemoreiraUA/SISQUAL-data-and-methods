import React, { useEffect, useState } from 'react';
import FastAPIClient from '../../client';
import config from '../../config';
import ClientsTable from "../../components/ClientsTable"
import DashboardHeader from "../../components/DashboardHeader";
import Footer from "../../components/Footer";
import Loader from '../../components/Loader';

const client = new FastAPIClient(config);

const Clients = () => {
     const [loading, setLoading] = useState(true)
     const [clients, setClients] = useState([])

     const [hasClients, setHasClients] = useState(false)

     useEffect(() => {
          fetchClients()
     }, [])

     const fetchClients = () => {
          setLoading(true)

          client.get_list_clients().then((data) => {
				setLoading(false)

				if (data?.clients.length > 0){
					setHasClients(true)
				}				
				
				setClients(data?.clients)
          }).catch((err) => {
			alert(err);
			console.log(err);
			setLoading(false);
          });
     }

     if (loading)
          return <Loader />

     return (
          <>
               <section
                   className="flex flex-col bg-white text-center"
                   style={{ minHeight: "100vh" }}
               >
                    <DashboardHeader />
                    <div className="container px-5 pt-6 mx-auto lg:px-20">
                        <div className="text-black">
							<nav className="shadow-lg bg-gray-100 mb-2 px-5 py-2 rounded-md w-full">
								<ol className="list-reset flex">
									<li>
										<span className="text-xl font-medium font-semibold">
											Registered Clients
										</span>
									</li>
								</ol>
							</nav>
                            <div>
							{hasClients && (
                                <ClientsTable
                                    clients={clients}
                                />
							)}
							{!hasClients && (
								<div
									onClick={(e) => {e.stopPropagation();}}
									onDoubleClick={(e) => {e.stopPropagation();}}
									onMouseDown={(e) => {e.stopPropagation();}}
									className="flex flex-wrap items-end justify-between w-full bg-white mb-1">
									<div className="w-full shadow-lg">
										No clients registered yet.

									</div>
								</div>		  
							)}
                            </div>
                        </div>
                    </div>
                    <Footer />
               </section>
          </>
     )
}

export default Clients;