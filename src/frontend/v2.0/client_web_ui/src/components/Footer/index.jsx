import React from 'react';
//import fctLogo from './2022_FCT_Logo_A_horizontal_branco.jpg';
import fctLogo from './logo_white.svg';
//className="fixed bottom-0 text-center bg-gray-200 h-15 w-full text-lg py-1"
function Footer() {
	return (
		<div 
			className="fixed bottom-0 text-center bg-gray-200 h-15 w-full text-lg py-1"
			style={{background: '#494963'}}
		>
			<footer className="lg:text-left">
				<div className="text-center p-2 text-white font-semibold">
					{/*POCI-01-0247-FEDER-039719*/}
					<a href="https://www.fct.pt/" rel="noreferrer noopener" target="_blank">
					<img 
						className="px-5"
						src={fctLogo} 
						alt="FCT - Fundação para a Ciência e a Tecnologia" />
					</a>
				</div>
			</footer>	  
		</div>
	);
}

export default Footer;