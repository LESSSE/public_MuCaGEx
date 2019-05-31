##LESSSE
##29 October 2018
##MuCaGEx
##____________
##Loading Datasets and Models metthods
##____________

import sys 
import os
if sys.version_info > (3,5,0):
    import importlib.util
elif sys.version_info > (3,3,0):
    from importlib.machinery import SourceFileLoader
elif sys.version_info > (2,0,0):
    import imp

def load_module(name,path):
    path = os.path.abspath(path)
    if sys.version_info > (3,5,0):
        spec = importlib.util.spec_from_file_location(name, path)
        dsconfig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dsconfig)
        return dsconfig
    elif sys.version_info > (3,3,0):
        dsconfig = SourceFileLoader(name, path).load_module()
        return dsconfig
    elif sys.version_info > (2,0,0):
        dsconfig = imp.load_source(name, path)
        return dsconfig

def load_dataset(path):
    dsconfig = load_module("ds",path+"/dsconfig.py")
    return dsconfig.dsconfig["load_dataset"]() 

def load_model(path):
    mconfig = load_module("m",path+"/mconfig.py")
    return mconfig.mconfig["load_model"]()

def get_dsconfig(path):
    dsconfig = load_module("ds",path+"/dsconfig.py")
    return dsconfig.dsconfig

def get_mconfig(path):
    mconfig = load_module("m",path+"/mconfig.py")
    return mconfig.mconfig
