from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from GlobalParams import GlobalParams
from src.classes.Enums import ClassificationProblem, Dataset
from src.classes.Serializable import Serializable
from src.classes.VariableList import VariableList


class ClassifierDataSet(Serializable):
    """Stores training or test data a classifier can be using"""

    @staticmethod
    def fromDict(dict:Optional[Dict]) -> Optional['ClassifierDataSet']:
        if dict is None: return None
        return ClassifierDataSet(
            dict["_noiseRed"],
            Dataset[dict["_combinationMethod"]],
            ClassificationProblem[dict["_classificationProblem"]],
            VariableList.fromDict(dict["vars"]),
            dict["X"],
            dict["y"],
            dict.get("plotIDs", None),
            dict.get("params", {}))

    def __init__(self, noiseRed:float, comMethod: Dataset, classificationProblem: ClassificationProblem,
                 vars: VariableList, X: np.array, y: np.array, plotIDs: np.array = None, params: Dict = None):
        self._noiseRed = noiseRed
        self._combinationMethod = comMethod
        self._classificationProblem = classificationProblem
        self._vars = vars
        self._X = X
        self._y = y
        self._plotIDs = plotIDs
        self._params = GlobalParams().__dict__
        if params is not None: self._params.update(params)

    def toDict(self, discardData: bool = False) -> Dict:
        """Converts the object to a dictionary
           :param discardData: If true the data is not included, only the variables and the problem"""
        return {
                "_noiseRed": self._noiseRed,
            "_classificationProblem": self._classificationProblem.name,
                "_combinationMethod": self._combinationMethod.name,
                "vars": self._vars.toDict(),
                "X": self._X.astype(np.float32) if not discardData else [],
                "y": self._y if not discardData else [],
                "plotIDs": self.plotIDs if not discardData else [],
                "params": self._params
                }

    def getStratifiedClassificationDataSplits(self, scale: bool = True, labelEncode: bool = True,
                                              testFolds=GlobalParams.testFolds, randomState=None):
        """Returns the test and training splits. Use randomState to get the same results, otherwise the results are randomized."""
        X, y = self._X, self._y
        scaler = None
        encoder = None
        if labelEncode:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        allTrains, allTests = [], []

        rkf = StratifiedKFold(n_splits=testFolds, shuffle=True, random_state=randomState)

        for train_index, test_index in rkf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            allTrains.append(ClassifierDataSet(self._noiseRed,self._combinationMethod, self._classificationProblem, self._vars, X_train, y_train))
            allTests.append(ClassifierDataSet(self._noiseRed,self._combinationMethod, self._classificationProblem, self._vars, X_test, y_test))

        return allTrains, allTests, scaler, encoder

    def getClassificationData(self, scale: bool = True, labelEncode: bool = True):
        """Returns the data in a format that can be used by a classifier"""
        X, y = self._X, self._y
        scaler = None
        encoder = None
        if labelEncode:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X,y,scaler,encoder
    def getClassificationDataSplit(self, scale: bool = True, labelEncode: bool = True,
                                   testSetSize: float = GlobalParams.testSetSize):
        """Returns the data in a format that can be used by a classifier"""
        X, y = self._X, self._y
        scaler = None
        encoder = None
        if labelEncode:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSetSize, stratify=y)

        trainData = ClassifierDataSet(self._noiseRed,self._combinationMethod,self._classificationProblem, self._vars, X_train, y_train)
        testData = ClassifierDataSet(self._noiseRed,self._combinationMethod,self._classificationProblem, self._vars, X_test, y_test)

        return trainData, testData, scaler, encoder

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def plotIDs(self) -> np.array:
        return self._plotIDs

    @property
    def numObs(self) -> int:
        return len(self._y)

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def classificationProblem(self) -> ClassificationProblem:
        return self._classificationProblem

    @property
    def combinationMethod(self) -> Dataset:
        return self._combinationMethod

    @property
    def noiseReduction(self) -> float:
        return self._noiseRed

    @property
    def vars(self) -> VariableList:
        return self._vars
