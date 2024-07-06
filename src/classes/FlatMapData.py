from typing import Union, Optional, TypeVar, Callable, Type

import numpy as np
from math import ceil, sqrt
from sklearn.preprocessing import StandardScaler

from src.__libs import pyutil
from src.classes.FileIDClasses import FileID
from src.classes.Serializable import Serializable, SerializableSubclass, FileIDSubclass
from src.classes.VariableList import VariableList
import paths as PATHS
import os
import pickle
import plotly.express as px


FlatMapSubclass = TypeVar('FlatMapSubclass', bound="FlatMapData")
class FlatMapData(Serializable):
    """

    Class is intended to be an abstract base class for similarityMaps, predictions, stacked data etc.

    Stores a 2D map data array that is NxMxF where NxM is the size of the map and F the number of features.
    The array is then flattened to NMxF and NaNs (majority of the pixels) are removed.
    Where the nans occurred is stored in the nanmask, this allows it to transform the prediction back into the original shape
    and display as a map.

    Each StackedData is specific to a year as climate is year dependent.
    """

    _data: np.ndarray
    _featureNames: list[str]
    _nanMask: np.ndarray
    _shape: tuple
    _fileID: FileIDSubclass

    def __init__(self, fileID:Optional[FileID], data:Union[np.ndarray, dict[str,np.ndarray]], featureNames:list[str], nanMask:np.ndarray = None, shape = (1476,1003), dtype = np.float32):
        super().__init__()
        self._nanMask = nanMask
        self._shape = shape
        self._featureNames = featureNames

        self._fileID = fileID
        self._data = data

        #this is a dummy
        if data is None:
            return
        #unfortunately 16 bit is not enough for some features
        if dtype is not None:
            self._data = self._data.astype(dtype)

        if nanMask is None: #create a nan mask
            # Remove the nans but store where they are - this saves space on the hard drive as nans are 70% of the 2D-data.
            self._nanMask = np.any(np.isnan(self._data), axis=1)
            self._data = self._data[~self._nanMask]
            assert np.any(np.isnan(self._data)) == False, "There are still nans in the data after removing them."


    def getDataAs1DArray(self, transform:Callable = None, subset:list[str] = None, mask1D:np.ndarray = None)->np.ndarray:
        ret = self._data.copy()
        if mask1D is not None:
            ret = ret[mask1D,:]

        if subset is not None:
            assert self.hasFeatures(subset), "The variables are not in the StackedData"
            ret = ret[:, [self._featureNames.index(v) for v in subset]]

        if transform is not None:
            ret = transform(ret)

        return ret

    def getDataAs2DArray(self, varSubset:Union[list[str],VariableList] = None)->np.ndarray:

        ret = self._data.copy()
        # pick out the right columns
        if varSubset is not None:
            if isinstance(varSubset, VariableList):
                varSubset = varSubset.list
            assert self.hasFeatures(varSubset), "The variables are not in the StackedData"
            ret = ret[:, [self._featureNames.index(v) for v in varSubset]]

        reshapedData = np.ones((len(self._nanMask),ret.shape[1])) * np.nan
        reshapedData[~self._nanMask,:] = ret
        return reshapedData.reshape(self._shape+(ret.shape[1],))

    def _getPlotTitle(self):
        return None

    def displayImg(self, data2D, normalized: bool = True, subsample:int = 1, returnOnly:bool = False, **layoutArgs):
        rows = ceil(sqrt(len(data2D)))
        cols = ceil(len(data2D) / rows)

        fullImg = np.zeros((self._shape[0] * cols, self._shape[1] * rows))
        for i, (k, val) in enumerate(data2D.items()):

            if normalized:
                val = (val - np.nanmin(val)) / (np.nanmax(val) - np.nanmin(val))

            fullImg[i // rows * self._shape[0]:(i // rows + 1) * self._shape[0],
            i % rows * self._shape[1]:(i % rows + 1) * self._shape[1]] = val

        if subsample > 1:
            fullImg = pyutil.subsampleAndAverage(fullImg, subsample)

        if returnOnly:
            return fullImg
        f = px.imshow(fullImg)

        # add annotations
        for i, (k, val) in enumerate(data2D.items()):
            f.add_annotation(y=(i // rows * self._shape[0] + self._shape[0] / 2)/subsample,
                             x=(i % rows * self._shape[1] + self._shape[1] / 2)/subsample,
                             text=k,
                             showarrow=False)

        f.update_layout(title=self._getPlotTitle(), **layoutArgs)

        f.show()

    def plotData(self, normalized: bool = True, featureSubset:list[str] = None, subsample:int = 1, mask:np.ndarray = None, returnOnly:bool = False, **layoutArgs):
        if featureSubset is None:
            featureSubset = self._featureNames

        data = self.getDataAsDict(lambda x: x in featureSubset, True)
        if mask is not None:
            for d,img in data.items():
                img[~mask] = np.nan
                data[d] = img

        f = self.displayImg(data, normalized, subsample,returnOnly, **layoutArgs)
        return f
    def getImage(self, normalized: bool = True, featureSubset:list[str] = None, subsample:int = 1):
        if featureSubset is None:
            featureSubset = self._featureNames

        return self.displayImg(self.getDataAsDict(lambda x: x in featureSubset, True), normalized, subsample, True)


    def getDataAsDict(self, featureNameFilter:Callable = None, reshape2D:bool = False):
        if featureNameFilter is not None:
            data = {k:self._data[:,i] for i,k in enumerate(self._featureNames) if featureNameFilter(k)}
        else:
            data = {k:self._data[:,i] for i,k in enumerate(self._featureNames)}

        if reshape2D:
            for k,v in data.items():
                reshapedData = np.ones_like(self._nanMask) * np.nan
                reshapedData[~self._nanMask] = v
                data[k] = reshapedData.reshape(self._shape)

        return data

    @property
    def defaultPath(self) -> str:
        return self._fileID.file

    def hasFeatures(self, vars:list[str])->bool:
        """Checks if the variables in Variable list are contained in this StackedData."""
        return len(set(vars) - set(self._featureNames)) == 0
    @property
    def shape(self)->tuple: return self._shape
    @property
    def nanmask(self)->np.ndarray: return self._nanMask

    @staticmethod
    def load(path, type: Type[FlatMapSubclass], idClass:Type[FileIDSubclass]) -> SerializableSubclass:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")

        with open(path, "rb") as f:
            data = pickle.load(f)
            return type.fromDict(data, type, idClass.parseFromPath(path))

    @staticmethod
    def fromDict(dict: dict, type:Type[FlatMapSubclass], fileID:FileIDSubclass) -> "Serializable":
        return type(fileID,
                           dict["data"],
                           dict["varList"],
                           dict["nanMask"],
                           dict["shape"]
                           )
    #Storing




    def saveToDisc(self, silent:bool = False):
        self._fileID.checkAndCreateSubfolder()
        with open(self._fileID.file, "wb") as f:
            pickle.dump(self.toDict(), f)
            if not silent:
                print(f"Written {self.__class__.__name__} to .../{self._fileID.fileName}")

    def toDict(self) -> dict:
        return {
            "data": self._data,
            "varList": self._featureNames,
            "nanMask": self._nanMask,
            "shape": self._shape
        }
