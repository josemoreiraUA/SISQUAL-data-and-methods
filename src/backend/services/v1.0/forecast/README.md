# Forecast Web Service

Forecast web service developed using [FastAPI](https://fastapi.tiangolo.com/).

## Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

Install Poetry using the Windows (Powershell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Notes:

 - You may need to add the directory where Poetry is installed to your `PATH` variable. 
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

If you get an error during this process, e.g., because of an incompatibility with the python version installed or used by default on your system you can proceed as follows:

If your system has a python 3.9 version installed run:

```bash
poetry env list
```

If a virtualenv exists, run:

```bash
poetry env remove app-duUi6HXW-py3.11
```

where `app-duUi6HXW-py3.11` is an example (your env should have a different name).

Then run:

```bash
poetry env use O:\Users\pc\.pyenv\pyenv-win\versions\3.9.13\python39.exe
```

where `O:\Users\pc\.pyenv\pyenv-win\versions\3.9.13\python39.exe` is the path to your python 3.9 executable.

Then run:

```bash
poetry env list
```

To make sure that a new `virtualenv` was created using `python 3.9`. It should have a name like, for example, `app-duUi6HXW-py3.9`, where `3.9` is the important part.

If your system does not have another python version installed, the process may get more complicated.

An example follows using [pyenv-win](https://github.com/pyenv-win/pyenv-win).

Install pyenv-win using the Windows (Powershell):

```bash
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

If you get an `UnauthorizedAccess` error then start Windows PowerShell with the `Run as administrator` option and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine`, then re-run the above installation command.

 - [Read 1](https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md#powershell)
 - [Read 2](https://windowsloop.com/enable-powershell-scripts-execution-windows-10/)

After installing pyenv-win, validate the installation:

```bash
pyenv --version
```

If getting an error: `The term pyenv is not recognized as the name of a cmdlet ...`

[Read](https://github.com/pyenv-win/pyenv-win/issues/97)

 - `O:\Users\pc\.pyenv\pyenv-win\bin` and `O:\Users\pc\.pyenv\pyenv-win\shims` should be declared in the PATH User and System variables before any declaration of a python installation.

where `O:\Users\pc\.pyenv\pyenv-win` should be replaced with the path on your system.

### Note:

For security reasons you may have to change your PowerShell execution policy back to `Restricted`.

 - Press the Windows Key to open the Start menu.
 - Type `Windows PowerShell`.
 - Right-click on the PowerShell result and select `Run as administrator`.
 - In the PowerShell window, execute `get-executionpolicy` to know the current execution policy being used.
 - Execute `set-executionpolicy Restricted`.
 - Execute `get-executionpolicy` to make sure the current execution policy is `Restricted`.

If you can run (may need to open a new console after the changes, `cmd` can be used):

```bash
pyenv --version
```

Run:

```bash
pyenv install -l
```

To get a list of the python versions supported. Choose a python 3.9 version and run:

```bash
pyenv install <python_version>
```

For example: `pyenv install 3.9.13`to install version `3.9.13`.

Check where the version was installed, e.g., `O:\Users\pc\.pyenv\pyenv-win\versions\3.9.13`.

Finally, run:

```bash
poetry env use O:\Users\pc\.pyenv\pyenv-win\versions\3.9.13\python39.exe
```

If the `virtualenv` was created run:

```bash
poetry install
```

## Running the web service

Move to the directory where the code for the app is `/app` and run:

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
 - [hypercorn](https://gitlab.com/pgjones/hypercorn) `link 1`
 - [hypercorn](https://pypi.org/project/hypercorn/) `link 2`
 - [hypercorn](https://hypercorn.readthedocs.io/en/latest/how_to_guides/api_usage.html) `link 3`
				  

### Web Service API documentation (needs to be connected to the internet):
 - http://127.0.0.1:8000/docs
 - http://127.0.0.1:8000/redoc
 - http://127.0.0.1:8000/openapi.json

I'm using port `8000` in these examples! Change this to the port being used!

## Web Service example output

`/api/v1/app/models/{model_id}/forecast`

{'forecast': [0, ..., 43], 'outparams': [0, 0, 0, 0, 0]}

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