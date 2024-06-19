from typing import Literal, Optional, Union

from GlobalParams import GlobalParams
from PathBase import *
from src.classes.Enums import ClassCombinationMethod, ClassificationProblem, ModelType, NoiseRemovalBalance
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
class Raw:
    # Raw occurrence data
    # _subfolder = "/NVS Data Aug 23"
    _subfolder = ""

    RecceData: str = DATAFOLDER + "_v2/_RawData" + _subfolder + "/Recce.csv"
    SaplingData = DATAFOLDER + '_v2/_RawData' + _subfolder + '/Saplings.csv'
    StemData = DATAFOLDER + '_v2/_RawData' + _subfolder + '/Stems.csv'
    SeedlingData = DATAFOLDER + '_v2/_RawData' + _subfolder + '/Seedlings.csv'

    # Raw Plot Info data
    SiteDescription = DATAFOLDER + '_v2/_RawData' + _subfolder + '/_SiteDescription_WithTP.csv'
    OrongoCoordinates = DATAFOLDER + '_v2/_RawData' + _subfolder + '/ORONGORONGO Plot Coordinates.csv'

    # Species Info
    SpeciesNames = DATAFOLDER + "_v2/_RawData/CurrentNVSNames_Raw.csv"

    # 2 - GIS DATA FOR PLOT INFO
    GisEcosystems = GISFOLDER + 'lris BasicEcosystems/'
    GisSoilLayers = GISFOLDER + 'Iris-SoilLayers/'
    GisNZEnvDSFolder = GISFOLDER + 'NZEnvDS_v1-1/'
    GisHotRunzFolder = GISFOLDER + 'HotRUNZ/'
    GisGRazingFolder = GISFOLDER + 'Grazing/'

    Predictors = f"_data/_input/_predictor_variables/"



class GISExport:
    TrainingDataSubset = GISFOLDER + "EftiExport/TrainingDataSubset.csv"

class Occ:
    # Merged full dataset
    Combined: str = "_data/_input/Occurences.csv"


class PlotInfo:
    # PLOT INFO - Contains things like coordinates, dates etc.  

    # Full plot info
    NoProps: str = "_data/_input/PlotInfo.csv"
    WithGeoProps: str = "_data/_input/PlotInfo_WithProps.csv"


class SpeciesInfo:
    Full = DATAFOLDER + "_v3/_TrainingData/1_SpeciesInfo.csv"


class Bioclim:
    AtPlotsByYear = '_data/_input/ClimateByYear.csv'
    AtPlotsByYearLinearAppx = '_data/_input/ClimateByYear_LinAppx.csv'


class Shifts:

    allPlotsByMethod =f'_data/_input/Migrations.csv'
    @staticmethod
    def allPlotsCombined(by: Union[str, ClassCombinationMethod]) -> str:
        if isinstance(by, ClassCombinationMethod):
            by = by.value

        return f'_data/_input/TrainingData_{by}.csv'


class Noise:

    possibleNoiseNetwork = DATAFOLDER + '_v3/_Analysis/NoiseNetwork.csv'

    @staticmethod
    def noisePerformance(comb:ClassCombinationMethod, vars:VariableList, problem:ClassificationProblem, balance:NoiseRemovalBalance)->str:
        return  DATAFOLDER + '_v3/_Models/NoisePerformance_%s_%s_%s%s.csv'%(comb.value,vars.name,problem.value,f"_{balance.value}" if balance != NoiseRemovalBalance.Combined else "")
    @staticmethod
    def noiseLabels(comb:ClassCombinationMethod, vars:VariableList)->str:
        return '_data/NoiseScores_%s_%s.csv'%(comb.value,vars.name)

class VarSelection:
    varSelectionBySpecies = DATAFOLDER + '_v3/_Analysis/1_VariableSelection_bySpecies_mrmr_scores_%s.csv'
    varSelectionTotal = DATAFOLDER + '_v3/_Analysis/1_VariableSelection_Total_mrmr_scores_%s.csv'

class Results:
    stackedFeaturesFolder = '_data/_predictions/_environment/'
    similaritiesFolder = '_data/_predictions/_similarity/'


    predictionsMeanFolder = DATAFOLDER + '_v3/_Analysis/MeanPredictions/'
    predictionsFolder = DATAFOLDER + '_v3/_Analysis/YearPredictions/'
    predictionImgsFolder = DATAFOLDER + '_v3/_Analysis/PredictionImgs/'
    permutationImportanceFolder = DATAFOLDER + '_v3/_Analysis/PermutationImportance/'
    permutationImportance = DATAFOLDER + '_v3/_Analysis/PermutationImportance/PermutationImportance.csv'
    gradientAnalysis = DATAFOLDER + '_v3/_Analysis/EFTIGradientAnalysis.csv'


    @staticmethod
    def eftiCorrAnalysis(comb:ClassCombinationMethod, name:str = "EFTICorrelationAnalysis"):
        s = DATAFOLDER + '_v3/_Analysis/%s_%s.csv'
        return s%(name,comb.value)
    @staticmethod
    def giniFeatureImportances(comb:ClassCombinationMethod, name:str = "GiniFI"):
        s = DATAFOLDER + '_v3/_Analysis/%s_%s.csv'
        return s%(name,comb.value)


class TrainedModels:

    ModelFolder = '_data/_models/'
