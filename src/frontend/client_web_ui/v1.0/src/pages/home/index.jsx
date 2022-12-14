import React, { useState } from "react";
import DashboardHeader from "../../components/DashboardHeader";
import Footer from "../../components/Footer";
import {Link} from 'react-router-dom';
import PopupModal from "../../components/Modal/PopupModal";
import FormInput from "../../components/FormInput/FormInput";
import Button from "../../components/Button/Button";
import FastAPIClient from "../../client";
import config from "../../config";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faTimesCircle } from '@fortawesome/free-solid-svg-icons/faTimesCircle'
import { faCheckCircle } from '@fortawesome/free-solid-svg-icons/faCheckCircle'

const client = new FastAPIClient(config);

function PopupModalMsg({ onCloseBtnPress, title, bColor, tColor, err, children }) {

	return (
		<div className="animate-fade-in-down container flex justify-center mx-auto">
			<div className="absolute inset-0 flex items-center justify-center bg-gray-700 bg-opacity-50"
                onClick={(e) => {
					e.preventDefault();
                }}
                onDoubleClick={(e) => {
					e.preventDefault();
                }}						
                onMouseDown={(e) => {
					e.preventDefault();
                }}>
				<div
					className="shadow-lg cursor-pointer top-0 w-full fixed rounded p-3 px-6 divide-blue-500"
					onClick={onCloseBtnPress}
					style={{ background : `${bColor}` }}
				>
					<div className="flex items-center justify-between">
						<h1 className="text-xl font-bold"
							style={{ color: `${tColor}` }}>
								{err && ( <> <FontAwesomeIcon icon={ faTimesCircle }/> {title} </>)}
								{!err && ( <> <FontAwesomeIcon icon={ faCheckCircle }/> {title} </>)}							
						</h1>
					</div>
					{children}
				</div>
			</div>
		</div>
	);
}

const Home = () => {
	
	const [showForm, setShowForm] = useState(false);
	const [error, setError] = useState({ id: "", culture: "" });
	
	const [clientParametersForm, setClientParametersForm] = useState({
		id: "",
		culture: ""
	});
	
	const [loading, setLoading] = useState(false);
	
	const [showClientMessage, setShowClientMessage] = useState(false);
	const [showMessage, setShowMessage] = useState(false);
	const [showErrorMessage, setErrorMessage] = useState(false);	
	
	const onUpdateParameters = (e) => {
		e.preventDefault();
		setLoading(true);
		setError(false);

		if (clientParametersForm.id.length <= 0) {
			setLoading(false);
			return setError({ id: "Define a client id." });
		}		
		
		if (clientParametersForm.culture.length <= 0) {
			setLoading(false);
			return setError({ culture: "Define a culture." });
		}

        const json_paylod = {'id': clientParametersForm.id,'culture': clientParametersForm.culture}

		client.create_client(
			json_paylod
		).then((data) => {
			setLoading(false);
			setShowForm(false);
			
			if (data?.detail == '1'){
                setShowClientMessage('Client created successfully!')
				setErrorMessage(false)
			}
			else{
                setShowClientMessage(data?.detail)
				setErrorMessage(true)
			}

			setShowMessage(true)
		}).catch((error) => {
			alert(error);
			console.log(error);
		});
	};
	
    return (
          <>
               <section
                   className="flex flex-col bg-white text-center"
                   style={{ minHeight: "100vh" }}
               >
                    <DashboardHeader />
                    <div className="container px-5 py-12 mx-auto lg:px-20">
						<div className="fixed bottom-20" style={{ overlay: {zIndex: 1} }}>

							<Link 
								to="/clients"
								className="inline-block px-6 py-3 bg-blue-400 text-white font-medium text-xs leading-tight uppercase rounded-full shadow-md hover:bg-blue-500 hover:shadow-lg focus:bg-blue-500 focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-600 active:shadow-lg transition duration-150 ease-in-out">
									Clients
							</Link>
							
							<span className='px-1'/>
							
							<button
								className="inline-block px-6 py-3 bg-gray-200 text-gray-700 font-medium text-xs leading-tight uppercase rounded-full shadow-md hover:bg-gray-300 hover:shadow-lg focus:bg-gray-300 focus:shadow-lg focus:outline-none focus:ring-0 active:bg-gray-400 active:shadow-lg transition duration-150 ease-in-out"
								onClick={(e) => {
									setShowForm(!showForm);
									setClientParametersForm({id: '', culture: ''});
									e.preventDefault();
								}}
							>
								Create Client
							</button>
						</div>			
                    </div>
		
                    <Footer />
               </section>

			{showMessage && !showErrorMessage && (
				<PopupModalMsg
					title={`${showClientMessage}`}
					bColor={'#161730'}
					tColor={'white'}
					err={false}
					onCloseBtnPress={(e) => {
						setShowMessage(false);
						e.preventDefault();
					}}
				/>
			)}
			{showMessage && showErrorMessage && (
				<PopupModalMsg
					title={`${showClientMessage}`}
					bColor={'#ff8730'}
					tColor={'white'}
					err={true}
					onCloseBtnPress={(e) => {
						//#f0bd84
						setShowMessage(false);
						e.preventDefault();
					}}
				/>
			)}
			{showForm && (
				<PopupModal
					title={"Create Client"}
					bColor={'#fcfeff'}
					tColor={'#0a4275'}					
					onCloseBtnPress={() => {
						setShowForm(false);
                        setError(false);
						setClientParametersForm({ ...clientParametersForm, culture: '' })
						setClientParametersForm({ ...clientParametersForm, id: '' })						
					}}
				>
					<div className="mt-4 text-left">
						<form className="mt-5" onSubmit={(e) => onUpdateParameters(e)}>
						
							<FormInput
								type={"text"}
								name={"id"}
								label={"Client Id"}
								placeholder={"client's id"}
								error={error.id}
								optional={false}
								value={clientParametersForm.id}
								onChange={(e) =>
									setClientParametersForm({ ...clientParametersForm, id: e.target.value })
								}
							/>						
							<FormInput
								type={"text"}
								name={"culture"}
								label={"Culture (Language)"}
								placeholder={"client's culture"}
								error={error.culture}
								optional={false}
								value={clientParametersForm.culture}
								onChange={(e) =>
									setClientParametersForm({ ...clientParametersForm, culture: e.target.value })
								}
							/>

							<Button
								loading={loading}
								error={error.culture}
								title={"Create Client"}
							/>
						</form>
					</div>
				</PopupModal>
			)}			   
          </>
    )
}

export default Home;