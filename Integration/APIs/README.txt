>>Ports<<
	PreProcesser: 8001
	FastForecaster: 8002 
	PostProcesser: 8003 (TODO)
	redis: 6379

>>PreProcesser<<
	Always gets the last 4 years of data (fixed but cant be changed in the future)

>>Running manually the APIs (CAREFUL: check last point)<<
	1 - The environment to run the APIs: conda activate api_deploy
	2 - To run the APIs use the following command inside of the /app dir: tox -e run
	3 - To run all tests only: tox

>>Celery & Redis<<
	Celery: is running permanently through the task scheduler
		(https://www.calazan.com/windows-tip-run-applications-in-the-background-using-task-scheduler/)
	Redis: Also running in background in Ubuntu through WSL1 
	(check if running with windows powershell: telnet localhost 6379)

>>APIs are also running in Windows Task Scheduler<<
	1 - If APIs src are updated then the task needs to be reloaded (sometimes they don't end properly, end python tasks in windows mngr)
	2 - In WinStart button search for task scheduler