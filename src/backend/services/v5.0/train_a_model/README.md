# Train Model Web Service

Train model web service developed using [FastAPI](https://fastapi.tiangolo.com/).

## Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

Install Poetry using the Windows (Powershell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Notes:

 - You may need to add the directory where Poetry is installed to your PATH variable. 
 - Check the Powershell console feedback after installing Poetry, for assistance.

## Setting up the project

Move to the directory where the `pyproject.toml` is and run:

```bash
poetry install
```

to install the dependencies and setup the `virtualenv`.

### Notes:

  - [Read](https://python-poetry.org/docs/cli/) for a detailled presentation of the commands provided by Poetry.
  - Use Poetry to add and remove dependencies, i.e., to manage dependencies!

  - If an error occurs during this step please read the installation notes for the `Forecast Service`.

## Running the web service

Move to the directory where the code for the app is `/app` and run:

```bash
poetry run hypercorn main:train_app -b 127.0.0.1:8001 --worker-class trio --workers 1
```

or

```bash
poetry run daphne -b 127.0.0.1 -p 8001 main:train_app
```

or

```bash
poetry run uvicorn main:train_app --workers 1 --host 127.0.0.1 --port 8001
```
			
to run rhe web service in a web server.		

### links:
 
 - [uvicorn](https://www.uvicorn.org)
 - [daphne](https://github.com/django/daphne)
 - [hypercorn](https://gitlab.com/pgjones/hypercorn) `link 1`
 - [hypercorn](https://pypi.org/project/hypercorn/) `link 2`
 - [hypercorn](https://hypercorn.readthedocs.io/en/latest/how_to_guides/api_usage.html) `link 3`
				  

### Web Service API documentation (needs to be connected to the internet):
 - http://127.0.0.1:8001/docs
 - http://127.0.0.1:8001/redoc
 - http://127.0.0.1:8001/openapi.json

I'm using port `8001` in these examples! Change this to the port being used!

## Web Service example output

...

## Managing DB migration and updates using [Alembic](https://alembic.sqlalchemy.org/en/latest/tutorial.html)

 - This service uses [SQLAlchemy](https://www.sqlalchemy.org) to connect and communicate with the database.
 - This service uses [SQLite](https://www.sqlite.org/index.html).
 
### Creating an Environment (The Migration Environment)

In the app directory `/app` run:

```bash
poetry run alembic init alembic
```

This will create several files and a new directory called `alembic` in `/app/alembic`.

 - Edit `alembic.ini` and `/app/alembic/env.py` as needed, according to your settings.
 
### Create a new revision

```bash
poetry run alembic revision --autogenerate -m "create initial tables"
```

### Run a new migration

```bash
poetry run alembic upgrade head
```

Alternatively, you can use a script to automatically create the initial tables

```bash
poetry run prestart.bat
```

 - This has already been done in the current project and the database is also provided.
 - These instructions can change in the future if another database engine is used.
 
## Usefull commands

Command | Description
--- | ---
`poetry show` | Shows the packages installed in the projects' virtualenv.
`poetry env list` | Lists the virtualenvs associated with the project.
`poetry add <module>` | Adds a dependency.
`poetry export --output requirements.txt` | Writes the projects' dependencies to a file.

## Contributing

...

## License

...
