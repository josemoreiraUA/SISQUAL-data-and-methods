# Forecast Web Service

Forecast web service developed using FastAPI.

## Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

Install Poetry using the Windows (Powershell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Notes:

 - You may need to add the directory where Poetry is installed to your PATH variable. 
 - Check the Powershell console feedback after installing Poetry, for assistance.

## Setting up the project

Move to the directory where the pyproject.toml is and run:

```bash
poetry install
```

to install the dependencies and setup the `virtualenv`.

### Notes:

  - [Read](https://python-poetry.org/docs/cli/) for a detailled presentation of the commands provided by Poetry.
  - Use Poetry to add and remove dependencies, i.e., to manage dependencies!

## Running the web service

Move to the directory where the code for the app is `*/app*` and run:

```bash
poetry run hypercorn app:app -b 127.0.0.1:8000 --worker-class trio --workers 1
```

or

```bash
poetry run daphne -b 127.0.0.1 -p 8000 app:app
```

or

```bash
poetry run uvicorn app:app --workers 1 --host 127.0.0.1 --port 8000
```
			
to run rhe web service in a web server.		

### links:
 
 - [uvicorn](https://www.uvicorn.org)
 - [daphne](https://github.com/django/daphne)
 - [hypercorn](https://gitlab.com/pgjones/hypercorn) link 1
 - [hypercorn](https://pypi.org/project/hypercorn/) link 2
 - [hypercorn](https://hypercorn.readthedocs.io/en/latest/how_to_guides/api_usage.html) link 3
				  

### Web Service API documentation (needs to be connected to the internet):
 - http://127.0.0.1:8000/docs
 - http://127.0.0.1:8000/redoc
 - http://127.0.0.1:8000/openapi.json

I'm using port 8000 in these examples! Change this to the port being used!

## Web Service example output

`/api/v1/app/models/{model_id}/forecast`

{'forecast': [0, ..., 43], 'outparams': [0, 0, 0, 0, 0]}

## Usefull commands

Command | Description
--- | ---
poetry show | Shows the packages installed in the projects' virtualenv.
poetry env list | Lists the virtualenvs associated with the project.
poetry add <module> | Adds a dependency.
poetry export --output requirements.txt | Writes the projects' dependencies to a file.

## Contributing

## License