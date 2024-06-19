import os.path
import pickle
from typing import Union, Optional

import pandas as pd
import numpy as np

from src.classes.FileIDClasses import ModelPredictionFileID
from src.classes.FlatMapData import FlatMapData
from src.classes.Serializable import Serializable
from src.classes.TrainedModel import TrainedModel
import paths as PATHS

class ModelPrediction(FlatMapData):
    """Stores Tru/False predictions by year"""

    def __init__(self,  data:Union[np.ndarray, dict[str,np.ndarray]], model:Optional[TrainedModel], years:list[int], nanMask:np.ndarray = None, shape = (1476,1003) ):
        mpfid = ModelPredictionFileID(years, model._fileID if model is not None else None)
        super().__init__(mpfid, data, [str(y) for y in years], nanMask, shape, np.bool_)

    def _getPlotTitle(self):
        return self._fileID["Species"]

    @staticmethod
    def load(path) -> "ModelPrediction":
        mmp = Serializable.load(path, ModelPrediction)
        mmp._fileID = ModelPredictionFileID.parseFromPath(path)
        return mmp

    def toDict(self) -> dict:
        d = super().toDict()
        return d

    @staticmethod
    def fromDict(dict: dict) -> "ModelPrediction":

        return ModelPrediction(dict["data"],
                                   None,
                                   dict["varList"],
                                   dict["nanMask"],
                                   dict.get("shape", (1476, 1003)),
                                   )