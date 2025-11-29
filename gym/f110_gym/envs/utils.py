import os 
import numpy as np 
import yaml 


def read_config(path):
    '''
    Helper function to load config file.

    Parameters
    ----------
    path: String
        Path to config file

    Return
    ------
    conf: dict
        Config file elements
    '''
    with open(path) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

def get_abs_path():
    '''
    Helper function to get parent directory of project.

    Parameters
    ----------
    None

    Return
    ------
    path: String
        Path to parent project directory.
    '''
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)

    return path
