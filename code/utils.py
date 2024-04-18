import numpy as np
import pandas as pd
import os
import pickle
import re


def save_object(obj, filename):
    # Overwrites any existing file.
    with open(os.path.join(os.path.dirname(os.getcwd()), "src", filename), 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(os.path.join(os.path.dirname(os.getcwd()), "src", filename), 'rb') as inp:
        obj = pickle.load(inp)
        return obj


def preprocess(v):
    return (v - v.min()) / (v.max() - v.min())


def MSE(y_true, y_pred, mask=None):
    if mask is None:
        return np.mean(np.square(y_true - y_pred), axis=-1)
    else:
        masked_diff = np.square(y_true - y_pred) * mask
        mask_sum = np.sum(mask, axis=-1)
        return np.sum(masked_diff, axis=-1) / mask_sum

def MAPE(y_true, y_pred):
    n = len(y_true)
    mape = (1/n) * sum([abs((y_true[i] - y_pred[i]) / y_true[i]) for i in range(n)]) * 100
    return mape

def MAE(y_true, y_pred, mask=None):
    if mask is None:
        return np.mean(np.abs(y_true - y_pred), axis=-1)
    else:
        masked_diff = np.abs(y_true - y_pred) * mask
        mask_sum = np.sum(mask, axis=-1)
        return np.sum(masked_diff, axis=-1) / mask_sum

# ================

def extract_last_word_from_filename(filename):
    # Используем регулярное выражение для поиска последнего слова в имени файла
    match = re.search(r'([^_]+)\.tsv$', filename)
    if match:
        return match.group(1)
    else:
        return None
    
def get_occur():
    
    filenames = []
    objectnames = []    

    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'ds003688', 'stimuli', 'annotations', 'sound')):
        for filename in files:
            filenames.append(os.path.join(root, filename))
            objectname = extract_last_word_from_filename(filename)
            objectnames.append(objectname)
            
    filenames.sort()
    objectnames.sort()
    
    occur = list()
    #objects = dict()
    #pairs = list()

    for filename, objectname in zip(filenames, objectnames):
        df = pd.read_csv(filename, sep='\t')
        df = (df * 25).astype(dtype=int)
        vector = np.zeros(9750, dtype=int)
        for ts in df.values:
            vector[ts[0]:ts[1]+1] = 1
        occur.append(vector)
        #objects[objectname] = {}
        #objects[objectname]['count'] = df.shape[0]
        #objects[objectname]['occurences'] = vector
        #pairs.append((objectname, df.shape[0]))
        
    occur = np.array(occur).T
    #pairs.sort(key=lambda x: x[1], reverse=True) # пары (объек, число появлений в кадре), отсортированные по убыванию числа появлений
   
    return occur