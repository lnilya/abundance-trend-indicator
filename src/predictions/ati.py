from itertools import product
from typing import List, Literal

import numpy as np
from src.classes.FileIDClasses import ModelFileID, ModelMeanPredictionFileID

from GlobalParams import GlobalParams
from src.__libs import mputil
from src.__libs.pyutil import termutil
from src.classes import VariableList
from src.classes.Enums import ModelType, ClassificationProblem, PredictiveVariableSet, NoiseRemovalBalance, \
    Dataset
from src.classes.ModelMeanPrediction import ModelMeanPrediction
from src.classes.ModelPrediction import ModelPrediction
from src.classes.Serializable import Serializable
from src.classes.StackedData import StackedData
from src.classes.TrainedModel import TrainedModel
import os
import paths as PATHS

def _pred(m:TrainedModel,ensembleHandling:Literal["hard","soft"], data):
    if ensembleHandling == "hard":
        return m.predict(data,False).reshape(-1, 1)

    return m.predict(data,True).reshape(-1, 1)


def computeATIPredictions(overwrite: bool, models: list[ModelType], varsets: list[VariableList],
                          noiseRed: list[float],
                          datasets:list[Dataset],
                          output:Literal["mean","single","both"] = "mean",
                          ensembleHandling:Literal["hard","soft"] = "soft",
                          yearRange = None,
                          speciesSubset = None):
    """
    Compute model predictions by gather all combinations of parameters and then running the entire prediction in parallel.
    :param overwrite: Will overwrite existing files, if false will skip the prediction for existing files
    :param models: List of models to generate predictions for
    :param varsets: List of variable sets to generate predictions for
    :param noiseRed: List of noise reduction values to generate predictions for
    :param datasets: List of datasets to generate predictions for
    :param output: mean, single or both. Mean will create average predictions over all years in the range (GlobalParams), single will create individual discrete predictions for each year.
    :param ensembleHandling: hard or soft. Hard will use the majority vote for the ensemble method to generate a binary label, soft will use the mean of the binary ensemble predictions resulting in a discrete number (e.g. 0, 0.25,0.5, ..).
    :return:
    """

    if not os.path.exists(PATHS.Results.predictionsMeanFolder) and output in ["mean","both"]:
        os.makedirs(PATHS.Results.predictionsMeanFolder)

    if not os.path.exists(PATHS.Results.predictionsFolder) and output in ["single","both"]:
        os.makedirs(PATHS.Results.predictionsFolder)


    Serializable.OMIT_WARNINGS = True

    yearRange = list(range(GlobalParams.yearRange[0], GlobalParams.yearRange[1])) if yearRange is None else yearRange

    # loadAll data
    data = {}
    for yr in yearRange:
        data[yr] = StackedData.readFromDisc(yr, PredictiveVariableSet.Full)


    for nr,m,var,dataset in product(noiseRed, models, varsets, datasets):
        termutil.chapPrint(f"[{m.value}] Predicting averages @ Noise {nr} ")
        print("")
        allModelFiles,_ = ModelFileID(NoiseRemovalBalance=NoiseRemovalBalance.Equal, NoiseReduction=nr, Classifier=m, Variables=var, Dataset=dataset, ClassificationProblem=ClassificationProblem.IncDec).getAllFiles()
        validModels:List[TrainedModel] = []
        #Check which models exist and need to be processed
        for mf in allModelFiles:
            classifier = TrainedModel.load(mf)
            if speciesSubset is not None and classifier.species not in speciesSubset:
                continue
            mmFile = ModelMeanPredictionFileID(yearRange,classifier._fileID)

            if mmFile.fileExists() and not overwrite:
                print( f"Skipping {allModelFiles[m]} on {classifier.modelID}. Already exists.")
                continue
            if classifier.testScore < GlobalParams.predictionThreshold:
                continue

            print(f"Queued {classifier.species} on {classifier.modelID.value} (Test: {classifier.testScore:.3f})")

            validModels.append(classifier)

        for i, v in enumerate(validModels):
            #Make prediction in parallel
            args = [(v,ensembleHandling,data[yr]) for yr in yearRange]
            #run predictions in parallel
            res = mputil.runParallel(_pred, args,GlobalParams.parallelProcesses,False, f"{v.species} on {v.modelID.value}[{GlobalParams.parallelProcesses}]")
            allRes = np.concatenate(res, axis=1)

            if output == "single" or output == "both":
                ms = ModelPrediction(allRes, v, yearRange, data[yearRange[0]].nanmask, data[yearRange[0]].shape)
                ms.saveToDisc()

            if output == "mean" or output == "both":
                m = ModelMeanPrediction(allRes, v, yearRange, data[yearRange[0]].nanmask, data[yearRange[0]].shape)
                m.saveToDisc()

            print(f"Completed {100*(i+1)/len(validModels):.2f}%")
            print("")


if __name__ == "__main__":
    computeModelPredictions(True, [ModelType.GLM], [PredictiveVariableSet.Full], [0.05], [NoiseRemovalBalance.Equal], Dataset.AdultsWithSameSplitByDBH, "single", "hard")
