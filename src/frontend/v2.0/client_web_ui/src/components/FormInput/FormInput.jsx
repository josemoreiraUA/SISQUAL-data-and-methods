import React from 'react';

function FormInput({label, optional=true, name, error, value, placeholder, onChange, type = "text"}) {
  return <div>
            <label 
				className={`${!optional ? "after:content-['*'] after:ml-0.5 after:text-red-500 " : ""} font-bold block mb-2 text-blue-500`}
				htmlFor={name}>
					{label}
			</label>
            <input
                type={type}
                name={name}
                value={value}
                placeholder={placeholder} 
                onChange={onChange}
                className={`rounded w-full p-2 border-b-2 ${!error ? "mb-6 border-blue-500 " : "border-red-500 "} text-teal-700 outline-none focus:bg-gray-300`}
            />
            {error && <span className='mb-3 text-red-500' >{error}</span>}
        </div>
}

export default FormInput;
