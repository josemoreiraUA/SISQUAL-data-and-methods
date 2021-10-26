Installation and script execution, step-by-step:

(1)
Install MinGW and msys -> all options by default + checkmark the following options in the installation manager: 
- MinGW developer toolkit;
- a basic MinGW installation (base tools); 
- GNU c++ compiler (gcc g++); 
- a basic MSYS installation;
URL: https://sourceforge.net/projects/mingw/files/latest/download

(Optional but recommended) Input into SYSTEM PATH: <path_to>\MinGW\msys\1.0\bin  

(2)
Install Miniconda 4.10.3 (py3.9) -> all options by default + checkmark the PATH option:
URL: https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Windows-x86_64.exe

(3)
execute the script with:
- bash <script_filename>.sh <environment_name>
The script attempts to take care of eventual errors, so wait until the end...