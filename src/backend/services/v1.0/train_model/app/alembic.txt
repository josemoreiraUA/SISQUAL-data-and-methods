#-------------------------------------------------------------------------------
  Managing DB migration and updates.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
  >> Using alembic and SQLAlchemy.

  1 - Creating an Environment (The Migration Environment).
  
  #---------
  Links:
  #---------
  
    *https://alembic.sqlalchemy.org/en/latest/tutorial.html
#-------------------------------------------------------------------------------
  in the app directory (/app) run:
  
	poetry run alembic init alembic

  #---------
  Notes:
  #---------

  This will create several files and a new directory called alembic in '/app/alembic'.

#-------------------------------------------------------------------------------
  2 - Edit alembic.ini and /app/alembic/env.py as needed, according to your settings.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
  3 - Create a new revision. (run in the dir where alembic.ini is located)
#-------------------------------------------------------------------------------

	poetry run alembic revision --autogenerate -m "create initial tables"

#-------------------------------------------------------------------------------
  4 - Run a new migration. (run in the dir where alembic.ini is located)
#-------------------------------------------------------------------------------

	poetry run alembic upgrade head

#-------------------------------------------------------------------------------
  5 - Use script to automatically create initial tables.
#-------------------------------------------------------------------------------

	poetry run prestart.bat

#-------------------------------------------------------------------------------