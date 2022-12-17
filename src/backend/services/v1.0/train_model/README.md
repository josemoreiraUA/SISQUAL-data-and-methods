#-------------------------------------------------------------------------------
# If using Poetry.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
  1 - Install Poetry using the Windows (Powershell):
#-------------------------------------------------------------------------------

	(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

  *See https://python-poetry.org/docs/#installing-with-the-official-installer for more information.

  #---------
  Notes:
  #---------

    - You may need to add the directory where Poetry is installed to your PATH variable. 
    - Check the Powershell console feedback after installing Poetry for assistance.

#-------------------------------------------------------------------------------
  >> Setting up the project.

  2 - Move to the directory where the pyproject.toml is and run:
#-------------------------------------------------------------------------------

	poetry install

  to install the dependencies and setup the virtualenv.

  *Read https://python-poetry.org/docs/cli/ for a detailled presentation of the commands provided by Poetry.

  #---------
  Notes:
  #---------

    - Use Poetry to add and remove dependencies, i.e., to manage dependencies.

#-------------------------------------------------------------------------------
  >> Running the web service.
  
  3 - Move to the directory where the code for the app is '/app' and run:
#-------------------------------------------------------------------------------

	poetry run hypercorn app:train_app -b 127.0.0.1:8001 --worker-class trio --workers 1

	or
	
	poetry run daphne -b 127.0.0.1 -p 8001 app:train_app
	
	or
	
	poetry run uvicorn app:train_app --workers 1 --host 127.0.0.1 --port 8001
			
  to run rhe web service in a web server.		

  #---------
  Notes:
  #---------
  
    - The web service can be executed, at least, in one of these web servers.
  
  #---------
  links:
  #---------
  
    - uvicorn		https://www.uvicorn.org

    - daphne		https://github.com/django/daphne

    - hypercorn 	https://gitlab.com/pgjones/hypercorn
					https://pypi.org/project/hypercorn/
					https://hypercorn.readthedocs.io/en/latest/how_to_guides/api_usage.html  

  #---------
  Web service API documentation (needs to be connected to the internet):

    - http://127.0.0.1:8001/docs
    - http://127.0.0.1:8001/redoc
	- http://127.0.0.1:8001/openapi.json

  I'm using port 8001 in these examples! Change this to the port being used!

#-------------------------------------------------------------------------------
  4 - Web service example output:
	
  
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
  5 - Usefull commands:
#-------------------------------------------------------------------------------

poetry show									shows the packages installed in the projects' virtualenv.

poetry env list								lists the virtualenvs associated with the project.

poetry add <module>							adds a dependency

poetry export --output requirements.txt		writes the projects' dependencies to a file.

#-------------------------------------------------------------------------------