from itertools import product

import pandas as pd

from GlobalParams import GlobalParams
from src.classes.Enums import ClassCombinationMethod, ClassificationProblem, PredictiveVariableSet, ModelType

from src.dataprocessing.geolayers import extractClimateLayers, extractGeoLayers, extractClimateLayersLinAppx
from src.datautil import datautil
import src.noisefilter as noise
import src.training as training
from src.shifts.shifts import identifyRemeasuredPlots
from src.shifts.shiftsAndClim import createTrainingData
from src.stacking.stackData import extractData
from src.predictions.similarity import computeSimilarity
import paths as PATHS

def run():

    # DEFINE PARAMETERS TO USE
    _datasets = [ClassCombinationMethod.AdultsOnly]
    _varsets = [PredictiveVariableSet.Full]
    _models = [ModelType.GLM, ModelType.SVM]
    _noiseReduction = [0,0.1,0.15]
    _overwrite = False
    _speciesSubset = ["Coprosma rotundifolia","Brachyglottis repanda"]


    #Step 1. Read the predictor variables at the plot coordinates and add them to the PlotInfo csv
    extractGeoLayers(_overwrite)
    extractClimateLayers(_overwrite)
    extractClimateLayersLinAppx(_overwrite)

    #Step 2. Identify the remeasured plots and create training data
    identifyRemeasuredPlots(_overwrite)
    for ds in _datasets:
        createTrainingData(True,ds)
    return

    if False:

        #Step 1. Compute the noise Scores - results stored in CSV files
        for (vs,ds) in product(_varsets,_datasets):
            noise.computeNoiseScores(_overwrite,ds,vs)

        #Step 2. Train Models - results stored in pickle files
        training.trainClassifiers(_overwrite,_models,_varsets,_datasets,_noiseReduction, _speciesSubset)

        #Step 3. Extract the predictor variables and format them into a format for prediction with the models
        for v in _varsets:
            extractData(GlobalParams.minYear, GlobalParams.maxYear, v)

    #Step 4. Calculatae the similarity between the training data and the entire space to be predicted
    for (vs,ds) in product(_varsets,_datasets):
        computeSimilarity(_overwrite,ds, vs)

    pass


if __name__ == '__main__':
    run()
