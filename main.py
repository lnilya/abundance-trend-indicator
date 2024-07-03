from itertools import product

import pandas as pd

from GlobalParams import GlobalParams
from src.classes.Enums import Dataset, ClassificationProblem, PredictiveVariableSet, ModelType

from src.dataprocessing.geolayers import extractClimateLayers, extractGeoLayers, extractClimateLayersLinAppx
from src.datautil import datautil
import src.noisefilter as noise
import src.training as training
from src.predictions.ati import computeATIPredictions
from src.shifts.shifts import identifyRemeasuredPlots
from src.shifts.shiftsAndClim import createTrainingData
from src.stacking.stackData import extractPredictionData
from src.predictions.similarity import computeSimilarity
import paths as PATHS

def run():

    #Parameters for the entire process. Generally each combination will be run in parallel, but the runtime increases with the number of combinations.
    _datasets = [Dataset.AdultsOnly] #Datasets to train on
    _varsets = [PredictiveVariableSet.Full] #Variable sets to train on
    _models = [ModelType.GLM] #Models to train
    _noiseReduction = [0,0.1] # Noise reduction to train on, each will produce an individual model with its own training set.
    _overwrite = True #If True will overwrite existing files
    _speciesSubset = ["Elevation Up Species","Precipitation Up Species"]

    if False:
        #Check GlobalParams.py for specific single parameters to be set (e.g. year period)
        #Step 1. Read the predictor variables at the plot coordinates and add them to the PlotInfo csv
        extractGeoLayers(_overwrite)
        extractClimateLayers(_overwrite)
        extractClimateLayersLinAppx(_overwrite)

    #Step 2. Identify the remeasured plots and create training data
    identifyRemeasuredPlots(_overwrite)
    for ds in _datasets:
        createTrainingData(_overwrite,ds)

    #Step 3. Compute the noise Scores - results stored in CSV files
    for (vs,ds) in product(_varsets,_datasets):
        noise.computeNoiseScores(_overwrite,ds,vs)

    #Step 4. Train Models - results stored in pickle files
    training.trainClassifiers(_overwrite,_models,_varsets,_datasets,_noiseReduction, _speciesSubset)

    #Step 5. Extract the variables and prepare them for prediction inside a stacked data object for each year
    # for v in _varsets:
    #     extractPredictionData(_overwrite, GlobalParams.minYear, GlobalParams.maxYear, v)

    #Step 6. Calculatae the similarity between the training data and the entire space to be predicted
    for (vs,ds) in product(_varsets,_datasets):
        computeSimilarity(True,ds, vs,speciesSubset=_speciesSubset)

    #Step 7. Use the models to make predictions for the entire space
    computeATIPredictions(_overwrite,_models, _varsets,_noiseReduction,_datasets,speciesSubset=_speciesSubset)



if __name__ == '__main__':
    run()
