# Time series forecasting using LSTM and neural networks

# Please, describe here the content of this folder in the following format
# date | author email | resource/dir name |
< description > 

# 2022/02/02 | pedrogusmao@ua.pt | Models Directory | 
<
The "Models" directory includes a variety of deep learning models that were researched such as:
	EncoderDecoder\ -> This directory has an initial implementation of a true Encoder-Decoder (Seq2Seq) 
		architecture that also uses tensorflow probabilites to generate confidence intervals, 
		this model is in a very raw state and was not very much explored. If it is of interest, 
		further development is required.
	
	Simple\ -> This directory includes multiple notebooks vanilla neural network models:
		1 - SeriesNet: check the repo -> https://github.com/kristpapadopoulos/seriesnet
		2 - SlidingWindow: a group of models that takes a number of lags of the time series in order to predict future steps
			a - LSTMs: Vanilla LSTMs, which has comments in build_model() that provide ways to increase its depth and 
				other configurations (e.g.: Many-to-Many);
			b - LSTMs_GapMonth: An implementation to check the performance of the model with gap months this is useful 
				for SISQUAL's use case (the difference is in the to_supervised() function);
			c - LSTMs_Simplified: Is a very simple example of how to forecast with TS all the complex 
				surrounding operations were removed
			d - MLP: A vanilla Multilayer Perceptron as it accepts a different input_shape
			e - StatefulLSTMs: LSTMs that use the stateful configuration. It is recommended that the input windows do not
				overlap and it is mandatory that the batch_size is equal on training and prediction phases.
				In order for it to function I did the following steps:
				1- in training the batch_size = highest common factor(p, H);
				2 - save the weights;
				3 - in prediction get the saved weights and predict with a different batch_size (e.g. 1)
			f - AutoRegressiveLSTMs: This is the sample implementation as the LSTMs notebook but it now uses the Recursive
				forecasting strategy.
		Also, all the models were greatly tested with univariate problems and even though I didn't encounter issues with
		multivariate/exogenous (whatever you want to call it) problems, I'd suggest further testing. These models are also prepared
		to work with datasets that were imputed ("Treatment" directory). Almost all variables are self-explanatory (any doubts contact me).

	Treatment\ -> This directory has notebooks that were made to treat the stores' datasets, these notebooks have methods to check the most
		frequent working schedules, schedule standardization, linear imputation and down/up sampling. 
		ScheduleTreatment_v2 (author: Bruno Mendes) also tries to standardize the number of days weekly considering days that the stores are closed. 

	Further work: Pipelining these entire workflows (sklearn has tools for that).

	Requirements: This work was done using conda environments and these are highly recommended. 
		In order to install the necessary libraries to run these demos check the "Integration\Env_Setup" folder and follow the exact steps given there 
		in the "installation_step_by_step_README.txt". This method assumes a windows OS. 
		If the shell script "NN_ENV_SETUP.sh" is causing problems you can open it and manually install the libraries specified there (conda/pip install). 
>