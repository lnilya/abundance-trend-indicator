from sklearn.preprocessing import StandardScaler

from src.classes.Serializable import Serializable
from sklearn.decomposition import PCA

from src.classes.VariableList import VariableList


class PCATransformation(Serializable):

    _pca: PCA
    _scaler: StandardScaler
    _varList: VariableList

    def __init__(self, pca:PCA, scaler:StandardScaler, varList:VariableList):
        self._pca = pca
        self._scaler = scaler
        self._varList = varList

    @property
    def explainedVariance(self)->float:
        return self._pca.explained_variance_ratio_
    @property
    def eigenValues(self)->float:
        return self._pca.explained_variance_


    def toDict(self) -> dict:
        return {
            "pca": self._pca,
            "scaler": self._scaler,
            "varList": self._varList.toDict()
        }

    @staticmethod
    def fromDict(dict: dict) -> "PCATransformation":
        return PCATransformation(dict["pca"],
                                    dict["scaler"],
                                 VariableList.fromDict(dict["varList"]))


