python ./backend_pre_start.py

alembic revision --autogenerate -m "create initial tables"

alembic upgrade head