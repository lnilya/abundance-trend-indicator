import os

import rasterio
import numpy as np
import pickle

from math import sqrt, ceil
from tqdm import tqdm

from GlobalParams import GlobalParams
from src.__libs import mputil
from src.__libs.osutil import getAllFiles
from src.__libs.plotutil.PlotlyGraph import PlotlyGraph
from src.classes.Serializable import Serializable
from src.classes.Enums import PredictiveVariableSet
import paths as PATHS
from src.classes.FileIDClasses import StackedDataFileID
from src.classes.StackedData import StackedData
from src.classes.VariableList import VariableList
from plotly import graph_objects as go
import plotly.express as px
import pandas as pd


def _readClimateVariables(vars:VariableList, year:int, nanvalue = (0,0)):

    res = {}
    shape = None
    for v in vars.list:
        #check if this is a directory
        dir = PATHS.Raw.Predictors + v + "/"
        if not os.path.isdir(dir): continue # not a climate variable, but static

        path = dir + f"{year}.tif"
        with rasterio.open(path) as src:
            rawData = src.read(1)
            if isinstance(nanvalue, tuple):
                nanvalue = rawData[nanvalue[0], nanvalue[1]]
            rawData[rawData == nanvalue] = np.nan
            shape = rawData.shape
            res[v] = rawData.reshape(-1)

    return res,shape

def _readGeoVariables(vars:VariableList, nanvalue = (0,0)):
    """
    :param vars: The vari
    :param nanvalue: GeoTiffs can load nanvalues as different values. It is usually a value like -3.4028235e+38 but can be something different. These MUST be replaced with NaNs during extraction.
    To replace all of these "nan-like" values there are two options: Provide an (x,y) tuple where you always expect a nan value of the geotiffs (for example the coordinates of a lake),
    or provide an actual value (e.g. -3.4028235e+38).
    :return:
    """
    #add all Geo Variables

    allVars = getAllFiles(PATHS.Raw.Predictors, "*.tif")

    allVars = {k:v for k,v in allVars.items() if v in vars.list}

    res = {}

    shape = None
    for path,name in allVars.items():
        with rasterio.open(path) as src:
            rawData = src.read(1)
            if name in GlobalParams.predictorVariableMappings.keys():
                mapping = GlobalParams.predictorVariableMappings[name]
                mapFunc = np.vectorize(lambda x: mapping.get(x, np.nan))
                rawData = mapFunc(rawData)
            else:
                #replace with nan values
                if isinstance(nanvalue, tuple):
                    nanvalue = rawData[nanvalue[0],nanvalue[1]]
                rawData[rawData == nanvalue] = np.nan

            shape = rawData.shape
            res[name] = rawData.reshape(-1)

    return res,shape


def _extractDataSingleYear(year, requiredVars:VariableList, geoVars):
    Serializable.OMIT_WARNINGS = True #Prevent pickle warnings
    climVars, shape = _readClimateVariables(requiredVars, year)
    climVars.update(geoVars)
    assert len(climVars) == len(requiredVars.list), f"Missing variables for year {year}"
    sfid = StackedDataFileID(year, requiredVars)
    sd = StackedData(sfid, climVars, requiredVars, None, shape)
    sd.saveToDisc()
    Serializable.OMIT_WARNINGS = False

def extractData(yearMin, yearMax, requiredVars:VariableList):
    """Compute the bioclim variables across entire new zealand. e.g. all pixels in hotrunz"""

    # 1. Load Elevation and Latitude data
    geoVars,shape = _readGeoVariables(requiredVars)
    mputil.runParallel(_extractDataSingleYear, [(year, requiredVars, geoVars) for year in range(yearMin, yearMax)], debug=False)

if __name__ == '__main__':
    extractData(GlobalParams.minYear, GlobalParams.maxYear, PredictiveVariableSet.Full)