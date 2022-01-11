# -*- coding: utf-8 -*-

import json
import settings as settings
from fbprophet.serialize import model_to_json, model_from_json

path, csvpath, imgpath, neg_values_path, large_file_path, model_path = settings.get_file_path()

# Save model to json file
def save_model(model, filaname='unnamed_model.json'):
    result='Model not saved!'
    with open(model_path+'/'+filaname, 'w') as fout:
        try:
            json.dump(model_to_json(model), fout)
            result='Model saved!'
            return result
        except:
            return result



def load_model(filename=None):
    result='Model not loaded!'
    if filename is None:
        raise Exception('Must inform Prophet model name to load')
    else:
        with open(model_path+'/'+filename, 'r') as fin:
            try:
                model = model_from_json(json.load(fin))
                return model
            except:
                return result
