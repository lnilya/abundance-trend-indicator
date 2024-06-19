from typing import Union, Optional, TypeVar, Callable, Dict

import numpy as np
from math import ceil, sqrt
from sklearn.preprocessing import StandardScaler

from src.classes.FileIDClasses import StackedDataFileID
from src.classes.FlatMapData import FlatMapData
from src.classes.Serializable import Serializable
from src.classes.VariableList import VariableList
import paths as PATHS
import os
import pickle
import plotly.express as px

class StackedData(FlatMapData):
    """
    For prediction every pixel on the NZ map is moved into an array and the 37 (or so features) measured.
    This yields a NxMxF array where where NxM is the size of the map and F the number of features.
    The array is then flattened to NMxF and NaNs (majority of the pixels) are removed.
    Where the nans occurred is stored in the nanmask, this allows it to transform the prediction back into the original shape
    and display as a map.

    Each StackedData is specific to a year as climate is year dependent.
    """

    def __init__(self, fileID:StackedDataFileID, data:Union[np.ndarray, dict[str,np.ndarray]], varList:VariableList, nanMask:np.ndarray = None, shape = (1476,1003) ):
        #stack the data into a numpy array if it comes as dict
        if isinstance(data, dict):
            # Final end result: For each year a list of 2D arrays sorted in the same way as variables, ready to be used for prediction
            dataAsList = [data[k] for k in varList]
            data = np.stack(dataAsList, axis=1)

        super().__init__(fileID,data, varList.list, nanMask, shape, np.float32)
        self._varList = varList


    def getDataAs1DArray(self, scale:StandardScaler = None, subset:VariableList = None)->np.ndarray:
        return super().getDataAs1DArray(scale.transform if scale is not None else None,subset.list if subset is not None else None)


    def plotData(self,  normalized: bool = True, subsample = 3):
        biofeatures = [f for f in self._featureNames if "BIO" in f]
        geofeatures = [f for f in self._featureNames if "BIO" not in f]
        if len(biofeatures) > 0: super().plotData(normalized, biofeatures,subsample)
        if len(geofeatures) > 0: super().plotData(normalized, geofeatures,subsample)



    @staticmethod
    def readFromDisc(year:int,var:VariableList)-> Optional["StackedData"]:
        fileID = StackedDataFileID(year, var)
        return StackedData.load(fileID.file)

    @staticmethod
    def checkIfExists(year:int,var:VariableList)->bool:
        return StackedDataFileID(year, var).fileExists()


    @staticmethod
    def load(path) -> "StackedData":
        return FlatMapData.load(path, StackedData, StackedDataFileID)

    @staticmethod
    def displayStackedDataBatch(year: int, vars: VariableList, normalized: bool = True):
        StackedData.readFromDisc(year, vars).plotData(normalized)

    @staticmethod
    def fromDict(dict: dict, type:any, fileID: StackedDataFileID) -> "StackedData":
        return StackedData(fileID,
                    dict["data"],
                    VariableList.fromDict(dict["varList"]),
                    dict["nanMask"],
                    dict["shape"]
                    )