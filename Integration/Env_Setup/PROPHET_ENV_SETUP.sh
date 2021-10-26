#!/bin/bash
CONDA=$(conda info --base)
CONDAPATH=${CONDA//\\//}	
cp "$CONDAPATH"/Library/bin/libcrypto-1_1-x64.* "$CONDAPATH"/DLLs
cp "$CONDAPATH"/Library/bin/libssl-1_1-x64.* "$CONDAPATH"/DLLs
conda create --yes --name $1 python=3.8.11
echo "This bash script will install a conda environment with the first argument as its name."
echo "All libraries required to run neural network models will be installed."
echo "Activating conda environment..."
source "$CONDAPATH"/etc/profile.d/conda.sh
conda config --env --set ssl_verify false
cp "$CONDAPATH"/envs/$1/Library/bin/libcrypto-1_1-x64.* "$CONDAPATH"/envs/$1/DLLs
cp "$CONDAPATH"/envs/$1/Library/bin/libssl-1_1-x64.* "$CONDAPATH"/envs/$1/DLLs
conda activate $1
echo "Installing libraries..."
conda install --yes -c conda-forge matplotlib=3.4.3
if test -f "$CONDAPATH"/pkgs/qt-5.12.9-h5909a2a_4/Scripts/.qt-post-link.bat ; then
	if grep -Fxq "set PATH=%PATH%;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;" "$CONDAPATH"/pkgs/qt-5.12.9-h5909a2a_4/Scripts/.qt-post-link.bat ; then
		echo "PATH seems OK!"
	else
		echo "Retrying matplotlib installation..."
		sed -i 's/@echo off/@echo off\nset PATH=%PATH%;%SystemRoot%\\system32;%SystemRoot%;%SystemRoot%\\System32\\Wbem;/' "$CONDAPATH"/pkgs/qt-5.12.9-h5909a2a_4/Scripts/.qt-post-link.bat
		conda install --yes -c conda-forge matplotlib=3.4.3
	fi
else 
	echo "Check later if matplotlib was properly installed."
fi
conda install --yes -c anaconda jupyter
if test -f "$CONDAPATH"/pkgs/qt-5.9.7-vc14h73c81de_0/Scripts/.qt-post-link.bat ; then
	if grep -Fxq "set PATH=%PATH%;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;" "$CONDAPATH"/pkgs/qt-5.9.7-vc14h73c81de_0/Scripts/.qt-post-link.bat ; then
		echo "PATH seems OK!"
	else
		echo "Retrying jupyter installation..."	
		sed -i 's/@echo off/@echo off\nset PATH=%PATH%;%SystemRoot%\\system32;%SystemRoot%;%SystemRoot%\\System32\\Wbem;/' "$CONDAPATH"/pkgs/qt-5.9.7-vc14h73c81de_0/Scripts/.qt-post-link.bat
		conda install --yes -c anaconda jupyter
	fi
else
	echo "Check later if jupyter was properly installed."
fi 
conda install --yes -c plotly plotly=5.1.0
conda install --yes -c conda-forge prophet=1.0.1
conda install --yes -c anaconda pandas=1.3.0
conda install --yes -c anaconda pyodbc=4.0.31
conda install --yes -c anaconda scikit-learn=0.23.2
conda install --yes -c anaconda statsmodels=0.12.0
conda install --yes -c conda-forge python-dotenv=0.19.0