import React from 'react';
//import fctLogo from './2022_FCT_Logo_A_horizontal_branco.jpg';
//import fctLogo from './logo_white.svg';
import projLogo from './logo.png';
import sisLogo from './sis.png';
import uaLogo from './ua.jpg';
//className="fixed bottom-0 text-center bg-gray-200 h-15 w-full text-lg py-1"
function Footer() {
	return (
		<div 
			className="fixed bottom-0 text-center bg-gray-200 h-15 w-full text-lg py-1"
			style={{background: '#FBFBFB'}}
		>
			<footer className="lg:text-left">
				<div className="text-center p-2 text-white font-semibold">
					{/*POCI-01-0247-FEDER-039719*/}

					<img 
						className="px-5"
						src={projLogo} 
						alt="FCT - Fundação para a Ciência e a Tecnologia"
						style={{display: 'inline'}} />

					<img 
						className="px-5"
						src={sisLogo} 
						alt="Sisqual Workforce Management" 
						style={{display: 'inline'}} />

					<img 
						className="px-5"
						src={uaLogo} 
						alt="Universidade de Aveiro"
						style={{display: 'inline'}} />

                    <br />

                    <span style={{color: '#030313'}}>
                        POCI-01-0247-FEDER-039719
                    </span>
					
				</div>
			</footer>	  
		</div>
	);
}

export default Footer;