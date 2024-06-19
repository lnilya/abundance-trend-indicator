import os.path
import pickle
from typing import Union, Optional

import pandas as pd
import numpy as np

from src.classes.FileIDClasses import ModelMeanPredictionFileID
from src.classes.FlatMapData import FlatMapData
from src.classes.Serializable import Serializable
from src.classes.TrainedModel import TrainedModel
import paths as PATHS
from src.classes.Enums import ClassCombinationMethod
from src.classes.VariableList import VariableList


class ModelMeanPrediction(FlatMapData):
    """Stores Tru/False predictions by year"""

    def __init__(self,data:Union[None,np.ndarray, dict[str,np.ndarray]], model:Optional[TrainedModel], years:list[int], nanMask:np.ndarray = None, shape = (1476,1003)):
        #average the predictions
        data = np.mean(data, axis=1, dtype=np.float32).reshape(-1,1) if data is not None else None
        if model is not None:
            featureName = f"{years[0]}-{years[-1]} | NR: {model.noiseReduction}"
            super().__init__(ModelMeanPredictionFileID(years,model._fileID),data, [featureName], nanMask, shape, np.float32)
        else:
            featureName = years[0] #when loading the feature name will be passed here
            super().__init__(ModelMeanPredictionFileID(years,None),data, [featureName], nanMask, shape, np.float32)

    def getDataAs2DArray(self, mask2D:np.ndarray = None) -> np.ndarray:
        """The mask is expected to have False where the value will be set to NaN """
        r = super().getDataAs2DArray(None)
        r = r[:,:,0]
        if mask2D is not None:
            r[~mask2D] = np.nan
        return r

    def _getPlotTitle(self): return self._fileID["Species"]

    @staticmethod
    def load(path) -> "ModelMeanPrediction":
        mmp = Serializable.load(path,ModelMeanPrediction)
        mmp._fileID = ModelMeanPredictionFileID.parseFromPath(path)
        y,n = mmp._fileID["Years"], mmp._fileID["NoiseReduction"]
        mmp._featureNames = [f"{y} | NR: {n}"]

        return mmp

    def toDict(self) -> dict:
        d = super().toDict()
        return d
    @staticmethod
    def fromDict(dict: dict) -> "ModelMeanPrediction":

        return ModelMeanPrediction(dict["data"],
                           None,
                           dict["varList"],
                           dict["nanMask"],
                           dict.get("shape",(1476,1003)),
                           )