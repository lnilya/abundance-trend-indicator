import os
import os.path
from collections import OrderedDict
from enum import Enum
from typing import Dict, Tuple, List, Literal
import shutil

import pandas as pd

from src.__libs import osutil
from src.classes import VariableList
from src.classes.Enums import ModelType, ClassificationProblem, NoiseRemovalBalance, ClassCombinationMethod, \
    PredictiveVariableSet
from src.classes.Serializable import Serializable
import paths as PATHS
import numpy as np

def _varlistToStr(e: VariableList): return e.name


def _enumToStr(e: Enum): return e.value


def _nop(e: str): return e


def _numToStr(f): return str(f)

def _neighboursToStr(f): return "N"+str(f)


def _yearsToStr(years: list[int]): return f"{years[0]}-{years[-1]}"

class FileID:
    # FOLDER: OrderedDict = OrderedDict(
    # )
    #
    CAST: OrderedDict = OrderedDict(
        Species=_nop,
        NumNeighbours=_neighboursToStr,
        Classifier=_enumToStr,
        Variables=_varlistToStr,
        ClassificationProblem=_enumToStr,
        ClassCombinationMethod=_enumToStr,
        NoiseReduction=_numToStr,
        NoiseRemovalBalance=_enumToStr,
        Years=_yearsToStr,
        Year=_numToStr,
    )

    _fileNameElements: List[List[str]]  #Last element is file name, before that are subfolders
    _idElements: Dict[str, str]

    def __init__(self, idElements: Dict, fileNameElements: List[List[str]], ext: str, basePath: str, skipCast:bool = False):

        #Remove none elements. If one of them is non, this instance can only be used to get all files matching this pattern
        if not skipCast:
            self._idElements = {}
            for k, v in idElements.items():
                if v is not None:
                    if isinstance(v,str):
                        self._idElements[k] = v
                    else:
                        self._idElements[k] = FileID.CAST[k](v)
        else:
            self._idElements = idElements

        self._fileNameElements = fileNameElements
        self._fileExtension = ext
        self._basePath = basePath

    @staticmethod
    def moveAndRename(basePath:str, newNameElements:List[List[str]], oldNameElements:List[List[str]],
                      ext:str, fillDefaults:Dict[str,str] = {},
                      op:Literal["copy","move"] = "move", newBasePath:str = None,
                      oldClass = None, newClass= None,
                      dryrun:bool = False, **filters):
        """Will load the files by the old names and move them to the scheme by the new names. If the new Name Elements do not contain all the id elements from oldNames
        they will be filled with fillDefaults."""
        if oldClass is None:
            oldFileID = FileID(filters, oldNameElements, ext, basePath)
        else:
            oldFileID = oldClass(**filters)
            oldFileID._fileNameElements = oldNameElements
            oldFileID._basePath = basePath


        allFiles,names = oldFileID.getAllFiles()

        #flatten oldNameElements
        flatOldNameElements = [el for sublist in oldNameElements for el in sublist]
        #remove all defaults that are in old elements. Otherwise they will be overwritten
        fillDefaults = {k:v for k,v in fillDefaults.items() if k not in flatOldNameElements}


        for path,ids in allFiles.items():
            ids = dict(zip(names,ids))
            #add defaults if needed
            #remove any keys in fillDefaults that are already in ids
            ids.update(fillDefaults)
            if newClass is None:
                newFileID = FileID(ids, newNameElements, ext, newBasePath if newBasePath is not None else basePath, True)
            else:
                newFileID = newClass(**ids)
                newFileID._fileNameElements = newNameElements
                newFileID._basePath = newBasePath if newBasePath is not None else basePath

            newFileID.checkAndCreateSubfolder()

            if not dryrun:
                if op == "move": shutil.move(path, newFileID.file)
                elif op == "copy": shutil.copy(path, newFileID.file)

            print(f"{op.capitalize()}")
            print(f"FROM {path}")
            print(f"TO {newFileID.file}")
            print("")




    def __getitem__(self, item):
        if item in self._idElements:
            return self._idElements[item]
        raise AttributeError(f"Attribute {item} not found in FileID")

    @staticmethod
    def parseFromPath(path:str, fileNameElements:List[List[str]]) -> "FileID":
        fileName = os.path.basename(path)

        #remove the extension
        ext = fileName.split(".")[-1]
        pathNoExt = path[:-len(ext)-1]
        basePath = "/".join(pathNoExt.split("/")[:-len(fileNameElements)]) + "/"
        idPath = pathNoExt.split("/")[-len(fileNameElements):]

        idPath = [i.split("_") for i in idPath]

        assert len(fileNameElements) == len(idPath), f"FileID parsing failed. Expected {len(fileNameElements)} elements, got {len(idPath)}"

        idElements = {}
        for i, fne in enumerate(fileNameElements):
            for j, el in enumerate(fne):
                idElements[el] = idPath[i][j]


        f = FileID(idElements, fileNameElements, ext, basePath, True)
        assert f.file == path, f"FileID parsing failed. Expected {path}, got {f.file}"
        return f

    def _getFilePattern(self) -> Tuple[str, List]:
        """Assembles the file and folder pattern ID and replaces missing values with *. Useful to get all files in a folder that match parts of the pattern."""
        fileID = []
        patternNames = []
        for layerElements in self._fileNameElements:
            layer = []
            for fe in layerElements:
                if fe in self._idElements:
                    layer += [self._idElements[fe]]
                else:
                    layer += ["*"]
                    patternNames += [fe]
            fileID += ["_".join(layer)]

        return ("/".join(fileID) + "." + self._fileExtension, patternNames)

    def getAllFiles(self) -> Tuple[Dict[str,List[str]], List]:
        pattern, placeholderNames = self._getFilePattern()
        return osutil.getAllFilesWithSubfolders(self._basePath, pattern), placeholderNames
    def getAllFilesAsDataframe(self) -> pd.DataFrame:
        pattern, placeholderNames = self._getFilePattern()
        files = osutil.getAllFilesWithSubfolders(self._basePath, pattern)
        df = pd.DataFrame(list(files.values()), columns=placeholderNames)
        df["File"] = list(files.keys())
        return df

    def fileExists(self) -> bool:
        """Checks if the file exists"""
        return os.path.exists(self.file)

    def checkAndCreateSubfolder(self):
        fname = self.fileName.split("/")
        if len(fname) == 1: return #no subfolders
        fname = fname[:-1]
        curFolder = self._basePath
        for subfolder in fname:
            #check if exists
            if not os.path.exists(curFolder + subfolder):
                os.mkdir(curFolder + subfolder)
            curFolder += subfolder + "/"

    @property
    def file(self) -> str:
        return self._basePath + self.fileName
    @property
    def fileName(self) -> str:
        """Assembles the file name inlcuding the subfolders used for IDing"""
        flatElements = set([el for sublist in self._fileNameElements for el in sublist])

        #check which elements are not in the union set
        els = set(self._idElements)

        assert els == set(flatElements), f"FileID is not fully defined. Missing elements: {flatElements - els}"

        return self._getFilePattern()[0]


class ModelMeanPredictionFileID(FileID):
    ELEMENTS = [["Years"],["ClassCombinationMethod", "ClassificationProblem"], ["Classifier", "Variables"],
                ["Species", "NoiseReduction", "NoiseRemovalBalance"]]

    BASEPATH = PATHS.Results.predictionsMeanFolder
    EXT = "pickle"

    def __init__(self, years: list[int] = None, model:"ModelFileID" = None):

        idEls:Dict[str,any] = {}
        if model is not None:
            #flatten
            flatElements = [el for sublist in ModelMeanPredictionFileID.ELEMENTS for el in sublist]
            idEls = {f: model[f] for f in flatElements if f in model._idElements}

        idEls["Years"] = years

        super().__init__(idEls,
                         ModelMeanPredictionFileID.ELEMENTS,
                         ModelMeanPredictionFileID.EXT,
                         ModelMeanPredictionFileID.BASEPATH)

    @staticmethod
    def parseFromPath(path: str) -> "FileID":
        return FileID.parseFromPath(path, ModelMeanPredictionFileID.ELEMENTS)
class ModelPredictionFileID(FileID):
    ELEMENTS = [["Years"],["ClassCombinationMethod", "ClassificationProblem"], ["Classifier", "Variables"],
                ["Species", "NoiseReduction", "NoiseRemovalBalance"]]

    BASEPATH = PATHS.Results.predictionsFolder
    EXT = "pickle"

    def __init__(self, years: list[int] = None, model:"ModelFileID" = None):

        idEls:Dict[str,any] = {}
        if model is not None:
            #flatten
            flatElements = [el for sublist in ModelMeanPredictionFileID.ELEMENTS for el in sublist]
            idEls = {f: model[f] for f in flatElements if f in model._idElements}

        idEls["Years"] = years

        super().__init__(idEls,
                         ModelPredictionFileID.ELEMENTS,
                         ModelPredictionFileID.EXT,
                         ModelPredictionFileID.BASEPATH)

    @staticmethod
    def parseFromPath(path: str) -> "FileID":
        return FileID.parseFromPath(path, ModelPredictionFileID.ELEMENTS)


class StackedDataFileID(FileID):
    ELEMENTS = [["Variables", "Year"]]
    BASEPATH = PATHS.Results.stackedFeaturesFolder
    EXT = "pickle"

    def __init__(self, year: int = None, variableList: VariableList = None):
        super().__init__(dict(
            Year=year,
            Variables=variableList),
            StackedDataFileID.ELEMENTS,
            StackedDataFileID.EXT,
            StackedDataFileID.BASEPATH)

    @staticmethod
    def parseFromPath(path: str) -> "FileID":
        return FileID.parseFromPath(path, StackedDataFileID.ELEMENTS)
class SimilarityDataFileID(FileID):
    ELEMENTS = [["Years"],["ClassCombinationMethod","Variables","NumNeighbours","Metric"], ["Species", "ClassificationProblem"]]
    BASEPATH = PATHS.Results.similaritiesFolder
    EXT = "pickle"

    def __init__(self, Years: list[int] = None, NumNeighbours:int = None, Metric:str = None, Species: str = None, Variables: VariableList = None,
                 ClassificationProblem: ClassificationProblem = None, ClassCombinationMethod: ClassCombinationMethod = None):
        super().__init__(dict(
            Years= Years,
            Metric = Metric,
            NumNeighbours=NumNeighbours,
            Species=Species,
            ClassificationProblem=ClassificationProblem,
            ClassCombinationMethod=ClassCombinationMethod,
            Variables=Variables),
            SimilarityDataFileID.ELEMENTS,
            SimilarityDataFileID.EXT,
            SimilarityDataFileID.BASEPATH)

    @staticmethod
    def initFromPrediction(mm:ModelMeanPredictionFileID, metric:str = None, numNeighbours:int = None):
        d = dict(mm._idElements)
        #delete unused elements from d
        for e in ["Classifier","NoiseReduction", "NoiseRemovalBalance"]: del d[e]

        d["Metric"] = metric
        d["NumNeighbours"] = numNeighbours

        return SimilarityDataFileID(**d)

    @staticmethod
    def parseFromPath(path: str) -> "FileID":
        return FileID.parseFromPath(path, SimilarityDataFileID.ELEMENTS)

class ModelFileID(FileID):
    ELEMENTS = [["ClassCombinationMethod", "ClassificationProblem"], ["Classifier", "Variables"], ["Species", "NoiseReduction", "NoiseRemovalBalance"]]
    BASEPATH = PATHS.TrainedModels.ModelFolder
    EXT = "pickle"

    def __init__(self, Species: str = None, Classifier: ModelType = None, Variables: VariableList = None,
                 ClassificationProblem: ClassificationProblem = None, ClassCombinationMethod: ClassCombinationMethod = None,
                 NoiseReduction: float = None, NoiseRemovalBalance: NoiseRemovalBalance = None):
        super().__init__(dict(
            Species=Species,
            Classifier=Classifier,
            Variables=Variables,
            ClassificationProblem=ClassificationProblem,
            ClassCombinationMethod=ClassCombinationMethod,
            NoiseReduction=NoiseReduction,
            NoiseRemovalBalance=NoiseRemovalBalance),
            ModelFileID.ELEMENTS,
            ModelFileID.EXT,
            ModelFileID.BASEPATH)

    @staticmethod
    def parseFromPath(path: str) -> "FileID":
        return FileID.parseFromPath(path, ModelFileID.ELEMENTS)

    def _getFilePattern(self) -> Tuple[str, List]:
        pattern,names = super()._getFilePattern()

        #NoiseRemovalBalance is not used if no noise is removed
        if "NoiseReduction" in self._idElements and self._idElements["NoiseReduction"] == 0:
            #Replace proportional and equal models with NoiseReduction 0 to "combined". These Are the same models.
            pattern = pattern.replace(NoiseRemovalBalance.Equal.value,NoiseRemovalBalance.Combined.value)
            pattern = pattern.replace(NoiseRemovalBalance.Proportional.value,NoiseRemovalBalance.Combined.value)

        return pattern,names

if __name__ == "__main__":

    ELEMENTS = [["ClassCombinationMethod","ClassificationProblem"],["Classifier","Variables"],["Species","NoiseReduction", "NoiseRemovalBalance"]]


    FileID.moveAndRename(PATHS.TrainedModels.ModelFolder,ELEMENTS,ModelFileID.ELEMENTS,"pickle",{},
                         op="copy",
                         newBasePath=PATHS.TrainedModels.ModelFolder2,
                         oldClass=ModelFileID, newClass=ModelFileID,
                         dryrun=True,Species="Pseudowintera colorata",Classifier=ModelType.GLM.value)

    # f = ModelFileID("species", ModelType.RF, PredictiveVariableSet.PC7,
    #                 ClassificationProblem.IncDec,
    #                 ClassCombinationMethod.AdultsWithSameSplitByDBH, 0.5, NoiseRemovalBalance.Combined)
    # ModelFileID.parseFromPath(f.file)