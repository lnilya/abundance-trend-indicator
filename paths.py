from typing import Literal, Optional, Union

from GlobalParams import GlobalParams
from PathBase import *
from src.classes.Enums import Dataset, ClassificationProblem, ModelType, NoiseRemovalBalance
from src.classes.VariableList import VariableList

"""
OCCURRENCE DATA - Contains presence of species in plots.
"""

plotFolder: str = REPOFOLDER + "_plots/"
plotFolderFigures: str = REPOFOLDER + "_plots/_figures/"
csvFolderFigures: str = REPOFOLDER + "_plots/_csvs/"

class Cache:
    SetSimilaritySpeciesCounts = REPOFOLDER + "_plots/SetSimilaritySpeciesCounts.csv"
    SetSimilarity:str = REPOFOLDER + "_plots/SetSimilarityCounts.csv"
    SetSimilarityRemData:str = REPOFOLDER + "_plots/SetSimilarityRem.csv"

    BufferFolder:str = REPOFOLDER + "_databuffer/"

    @staticmethod
    def toPickle(filename:str)->str:
        return REPOFOLDER + '_plots/%s.pickle'%filename
class Virtual:
    PlotDistribution = f"_data/NZPlotDistribution.pickle"
    VirtualPlotInfoWithProps = f"_data/_virtual/VirtualPlotInfoWithProps.csv"

class Raw:

    Predictors = f"_data/_input/_predictor_variables/"



class Occ:
    # Merged full dataset
    Combined: str = "_data/_input/Occurrences.csv"


class PlotInfo:
    # PLOT INFO - Contains things like coordinates, dates etc.  

    # Full plot info
    NoProps: str = "_data/_input/PlotInfo.csv"
    WithGeoProps: str = "_data/_trainingdata/PlotInfo_WithProps.csv"


class SpeciesInfo:
    Full = DATAFOLDER + "_v3/_TrainingData/1_SpeciesInfo.csv"


class Bioclim:
    AtPlotsByYear = '_data/_trainingdata/ClimateByYear.csv'
    AtPlotsByYearLinearAppx = '_data/_trainingdata/ClimateByYear_LinAppx.csv'


class Shifts:

    allPlotsByMethod =f'_data/_trainingdata/Migrations.csv'
    @staticmethod
    def allPlotsCombined(by: Union[str, Dataset]) -> str:
        if isinstance(by, Dataset):
            by = by.value

        return f'_data/_trainingdata/TrainingData_{by}.csv'


class Noise:

    @staticmethod
    def noiseLabels(comb:Dataset, vars:VariableList)->str:
        return '_data/_trainingdata/NoiseScores_%s_%s.csv'%(comb.value,vars.name)


class Results:
    stackedFeaturesFolder = '_data/_predictions/_environment/'
    similaritiesFolder = '_data/_predictions/_similarity/'


    predictionsMeanFolder = '_data/_predictions/_ati/'
    predictionsFolder = '_data/_predictions/_ati_yearly/'

    @staticmethod
    def eftiCorrAnalysis(comb:Dataset, name:str = "EFTICorrelationAnalysis"):
        s = DATAFOLDER + '_v3/_Analysis/%s_%s.csv'
        return s%(name,comb.value)
    @staticmethod
    def giniFeatureImportances(comb:Dataset, name:str = "GiniFI"):
        s = DATAFOLDER + '_v3/_Analysis/%s_%s.csv'
        return s%(name,comb.value)


class TrainedModels:

    ModelFolder = '_data/_models/'
