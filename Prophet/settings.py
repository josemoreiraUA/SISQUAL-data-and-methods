# settings.py
import sys
import os
from os.path import join, dirname
from dotenv import load_dotenv, find_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(find_dotenv())

settings = {
    'PATH': dotenv_path,
    'CSVPATH': os.environ.get("CSVPATH"),
    'IMGPATH': os.environ.get("IMGPATH"),
    'NEGVALPATH': os.environ.get("NEGVALPATH"),
    'LARGCSVPATH': os.environ.get("LARGCSVPATH"),
    'MODELPATH': os.environ.get("MODELPATH"),
    'DBNAME': os.environ.get("DBNAME"),
    'DBUSER': os.environ.get("DBUSER"),
    'DBPASS': os.environ.get("DBPASS"),
    'DBHOST': os.environ.get("DBHOST"),
    'DBPORT': os.environ.get("DBPORT")
}

def get_file_path():
    linux_plat = ['linux','linux1', 'linux2']
    if sys.platform in linux_plat:
        path = str(settings.get('PATH'))[:-5]
        csvpath = path + str(settings.get('CSVPATH'))[-15:]
        imgpath = path + str(settings.get('IMGPATH'))[-4:]
        neg_values_path = imgpath + str(settings.get('NEGVALPATH'))
        large_file_path = path + str(settings.get('LARGCSVPATH'))[-4:]
        model_path = path + str(settings.get('MODELPATH'))[-7:]
        return path, csvpath, imgpath, neg_values_path, large_file_path, model_path
    elif sys.platform == 'darwin':
        path = str(settings.get('PATH'))[:-5]
        csvpath = path + str(settings.get('CSVPATH'))
        imgpath = path + str(settings.get('IMGPATH'))
        neg_values_path = imgpath + str(settings.get('NEGVALPATH'))
        large_file_path = path + str(settings.get('LARGCSVPATH'))[-6:]
        model_path = path + str(settings.get('MODELPATH'))[-8:]
    elif sys.platform == 'win32':
        pass
     # Todo do something specific to windows operating system

    return path, csvpath, imgpath, neg_values_path, large_file_path, model_path


def get_acces_db_data():
    DBNAME = settings.get('DBNAME')
    DBUSER = settings.get('DBUSER')
    DBPASS = settings.get('DBPASS')
    DBHOST = settings.get('DBHOST')
    DBPORT = settings.get('DBPORT')
    return DBNAME, DBUSER, DBPASS, DBHOST, DBPORT
