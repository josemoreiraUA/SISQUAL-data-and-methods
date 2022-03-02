import sys
import os
from dotenv import load_dotenv

env_path=os.path.join('env_var', '.env')
#dotenv_path = join(dirname(__file__), '.env')

load_dotenv(env_path)

settings = {
	'SERVER': os.getenv('SERVER'),
	'DATABASE': os.getenv('DATABASE'),
	'DATABASE_SCHEDULE': os.getenv('DATABASE_SCHEDULE'),
	'USERNAME': os.getenv('USERNAME'),
	'PASSWORD': os.getenv('PASSWORD')
}

def get_access_db_data():
	SERVER = settings.get('SERVER')
	DATABASE = settings.get('DATABASE')
	DATABASE_SCHEDULE = settings.get('DATABASE_SCHEDULE')
	USERNAME = settings.get('USERNAME')
	PASSWORD = settings.get('PASSWORD')

	return SERVER, DATABASE, DATABASE_SCHEDULE, USERNAME, PASSWORD

#print(settings.get('SERVER'))