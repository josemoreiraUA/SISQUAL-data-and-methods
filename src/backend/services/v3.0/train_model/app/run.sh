#!/bin/sh

# run with
# poetry run ./run.sh

#export APP_MODULE=${APP_MODULE-app.main:app}
export APP_MODULE=${APP_MODULE-app:train_app}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-8001}

#exec uvicorn --reload --host $HOST --port $PORT "$APP_MODULE"
exec uvicorn --workers 1 --host $HOST --port $PORT "$APP_MODULE"