from enum import Enum
from typing import TypeVar, List

import pandas as pd

from src.classes.VariableList import VariableList

class NoiseRemovalBalance(Enum):
    # Given a fixed noise removal rate D we do...
    Equal = "equal" #...remove D% from each class
    Proportional = "proportional" #...remove D1 and D2 from the classess where D1/D2 = classSize1/classSize2. Smaller class looses less points
    Combined = "combined" #...remove D% from the entire dataset. Can completely focus on a single class, or both depending on noise score.

class PointDifficulty(Enum):

    Unlearnable = "Unlearnable"
    Learnable = "Learnable"
    Trivial = "Trivial"
    TrLe = "Trivial + Learnable"
    All = "All"

    @staticmethod
    def EMHMapping(df:pd.DataFrame = None, col:str = None):
        mapping = {"E":PointDifficulty.Trivial.value,"M":PointDifficulty.Learnable.value,"H":PointDifficulty.Unlearnable.value, "E +M":PointDifficulty.TrLe.value,
                   "Easy":PointDifficulty.Trivial.value,"Medium":PointDifficulty.Learnable.value,"Hard":PointDifficulty.Unlearnable.value, "Easy +Medium":PointDifficulty.TrLe.value}
        if df is not None:
            df[col] = df[col].map(mapping)
        return df, mapping


class ModelType(Enum):
    """Type of model being used"""
    RF = "RF" #Random Forest
    SVMW = "SVMW" #Support Vector Machine with balanced class weights
    GLM = "GLM" #Generalized Linear Model
    ANN1 = "ANN1" #Neural Network with 1 hidden layer

    KNN = "KNN"
    SVM = "SVM"
    SVMPoly = "SVMPoly"
    SVMSig = "SVMSig"
    SVMProb = "SVMProb"
    LSKNN = "LSKNN"
    LSRBF = "LSRBF"
    GPM = "GPM"
    GP = "GP"
    ANN2 = "ANN2"

    ## Those can't be trained but are the result of the training. Since training error is estimated through 4 fold cross validation
    # The result are 4 models. An option is to use these 4 models in an ensemble fashion.
    # Alternatively retrain the entire model on the full dataset. (these retrained models will retain the original SVM enum label)
    SVMEnsemble = "SVMEnsemble" #Support Vector Machine
    SVMWEnsemble = "SVMWEnsemble" #Support Vector Machine
    RFEnsemble = "RFEnsemble" #Support Vector Machine
    GLMEnsemble = "GLMEnsemble" #Support Vector Machine
    ANN1Ensemble = "ANN1Ensemble" #Support Vector Machine
    ANN2Ensemble = "ANN2Ensemble" #Support Vector Machine

    def parseToNonEnsembleVersion(self):
        if "Ensemble" in self.value:
            return ModelType(self.value.replace("Ensemble",""))
        return self
    def parseToEnsembleVersion(self):
        if "Ensemble" not in self.value:
            return ModelType(self.value + "Ensemble")
        return self

class ClassCombinationMethod(Enum):
    """We get an inc/dec in seedlings/saplings/stems or cover. These can be combined differently.
    For example we might only be interested in the increase/decrease of stems or we can think of a combined
    method that needs to have stems and seedlings or saplings increase."""

    AdultsOnly = "AdultsOnly"
    AdultsWithSameSplitByDBH = "AdultsWithSameSplitByDBH" #Same becomes inc/dec if maximum diameter increased or decreased

class ClassLabels(Enum):
    """Output of the predictor. These will be encoded with a LabelEncoder provided with the model."""

    Rem = "R"
    Add = "A"
    Inc = "I"
    Dec = "D"
    Same = "S"
    IncSame = "I+S"
    DecSame = "D+S"


class ClassificationProblem(Enum):
    """Defines which classification problem is solved or which training labels we have.
    Generally we have 3 classes: Increase, Decrease, Same. or a 3 class classification problem.
     But we can combine them into 1vs2 classification problems for various purposes.
     """
    DecIncSame = "DecIncSame"
    IncRest = "IncRest"
    DecRest = "DecRest"
    IncSame = "IncSame"
    DecSame = "DecSame"
    IncDec = "IncDec"

    _tv = TypeVar("ClassificationType",bound=[str,ClassLabels])
    def toLabels(self,type:_tv)->List[_tv]:
        if self == ClassificationProblem.IncSame:
            r = [ClassLabels.Inc, ClassLabels.Same]

        elif self == ClassificationProblem.DecSame:
            r = [ClassLabels.Dec, ClassLabels.Same]

        elif self == ClassificationProblem.IncDec:
            r = [ClassLabels.Inc, ClassLabels.Dec]
        else:
            r = [ClassLabels.Dec, ClassLabels.Inc, ClassLabels.Same]

        if type == str:
            r = [x.value for x in r]

        return r




class PredictiveVariableSet:

    Full = VariableList("GEOAllBioEndGr2", ["WindExposition", "Slope", "Ruggedness", "Roughness", "Elevation", "pH", "Longitude", "Latitude", "Drainage", "DistanceToTheCoast", "DistanceToNearestRiver", "CalciumContent", "Aspect Eastness", "Aspect Northness", "DistanceToTheRoad"] + [f"BIOEnd{i}" for i in range(1, 20)])
    PC7 = VariableList("PC7", [f"PC{d}" for d in list(range(1,8))])
    MinCorrelated = VariableList("MinCorr", [ "DistanceToNearestRiver", "Aspect Eastness", "Aspect Northness", "Elevation", "Latitude", "BIOEnd12", "Slope", "pH", "BIOEnd1", "WindExposition", "BIOEnd3", "BIOEnd7", "BIOEnd15",]) #MAT, MTCO, Elevation and Latitude

    @staticmethod
    def fromString(s:str)->VariableList:

        #Get by name
        obj = getattr(PredictiveVariableSet, s, None)
        if isinstance(obj,VariableList):
            return obj

        #Get by ID... since sometimes diferent
        #search through all attributes that are instances of VariableList
        for v in vars(PredictiveVariableSet):
            obj = getattr(PredictiveVariableSet,v)
            if isinstance(obj,VariableList):
                if obj.name == s:
                    return obj