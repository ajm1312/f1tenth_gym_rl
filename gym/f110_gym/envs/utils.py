import os 
import numpy as np 
import yaml 



def read_config(path):
    with open(path) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

def get_abs_path():
    # 1. Get the absolute path of the current script
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)

    return path
