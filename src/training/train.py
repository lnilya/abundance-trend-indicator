from itertools import product
from time import sleep

import numpy as np
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

import src.training.classifierDefinitions as defs
from GlobalParams import GlobalParams
from src.__libs.pyutil import termutil
from src.classes.ClassifierDataSet import ClassifierDataSet
from src.classes.Enums import ClassificationProblem, PredictiveVariableSet, ModelType, Dataset, \
    NoiseRemovalBalance
from src.classes.PretrainedVotingClassifier import PretrainedVotingClassifier
from src.classes.TrainedModel import TrainedModel
from src.classes.Types import SpeciesName
from src.datautil.datautil import loadTrainingData
import paths as PATHS
import os

def _printSingleResult(testScores,trainScores,m:TrainedModel, fullTrainData:ClassifierDataSet):
    meanTest = np.mean(testScores)
    meanTrain = np.mean(trainScores)
    testScores = [f"{t:.2f}" for t in testScores]
    trainScores = [f"{t:.2f}" for t in trainScores]
    #wait 0.1 s
    print(
        f"         {m.modelID.value.ljust(3, ' ')} on {m.species}({len(fullTrainData.y)} observations) -> Test: {meanTest:.3f} Train: {meanTrain:.3f}")
    sleep(0.01)



def trainClassifiers(overwrite:bool = True,
                     _models = (ModelType.GLM, ModelType.SVM, ModelType.RF),
                     _varsets = (PredictiveVariableSet.PC7, PredictiveVariableSet.Full, PredictiveVariableSet.MinCorrelated),
                     _datasets = (Dataset.AdultsWithSameSplitByDBH,),
                     _noiseReduction = (0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175),
                     speciesSubset:list[str] = None, saveModelTrainingData = False):

    """
    Trains classifiers for all species in the dataset. The classifiers are trained on the training data and the best hyperparameters are determined using a grid search.
    :param overwrite: If true, will overwrite existing models. If false, will skip the training of models that already exist.
    :param _models: The models to use for training
    :param _varsets: The varable sets to use for training
    :param _datasets: The datasets to use for training (e.g. training data)
    :param _noiseReduction: Values of noise reduction to train on.
    :param speciesSubset: If not None, only train models for the species in this list
    :param saveModelTrainingData: If true will store the training data inside the model file, otherwise will only store the model itself.
    :return: None, results stored in TrainedModel objects on disk
    """

    #create output folder
    if not os.path.exists(PATHS.TrainedModels.ModelFolder):
        os.makedirs(PATHS.TrainedModels.ModelFolder)

    _classificationProblems = [ClassificationProblem.IncDec]

    _randomStateForSplits = 99 #The random state for the splits should be fixed.
    # This way all modes, variables and noise reductions are tested on the same splits. If the length is the same.
    # While it introduces some bias, the results become more comparable. Since the classification variance can be high this is necessary.

    res = []
    for v,c,nr,comb in product(list(_varsets), list(_classificationProblems), list(_noiseReduction), list(_datasets)):

        termutil.chapPrint(f"Training classifiers for {v.name} - {c.value} - Noise Cutoff: {nr if nr is not None else 0:.2f} ")

        dt:dict[SpeciesName,ClassifierDataSet] = loadTrainingData(comb, c, v, False,False,nr,False,NoiseRemovalBalance.Equal)

        if speciesSubset is not None:
            dt = {k:v for k,v in dt.items() if k in speciesSubset}

        numSpecies = len(dt.keys())

        for i,species in enumerate(dt.keys()):
            #Create multiple training-test splits
            trainDatas, testDatas, s, le = dt[species].getStratifiedClassificationDataSplits(randomState=_randomStateForSplits)
            termutil.infoPrint(f"Training {species}:")
            for classifier in _models:
                clf = defs.getClassifier(classifier,v)
                res.append(_trainSingleModel(dt[species],species,classifier,clf,not overwrite,trainDatas,testDatas,s,le,nr, saveModelTrainingData))
            termutil.successPrint(f"Completed {species} ({i+1}/{numSpecies}, {100*(i+1) / numSpecies:.2f} %)")
            print("")

    return res



def _trainSingleModel(cdf:ClassifierDataSet, species:str, classifier:ModelType, clf:GridSearchCV, _skipExistantResults:bool, trainDatas:list[ClassifierDataSet], testDatas:list[ClassifierDataSet], scaler, labelEncoder, noiseReduction:float = 0, saveData = False):
    """Trains a single model (with optimal hyperparameters). And stores it/scores in the scoresDF"""
    if _skipExistantResults:

        model = TrainedModel(species, classifier, cdf, trainedClassifier=clf,noiseReduction=noiseReduction,noiseBalance=NoiseRemovalBalance.Equal)

        if model.fileExists():
            print(f"         Skipping {species} on {classifier} as model file is already present")
            return

    try:

        ret = []
        testScores, trainScores = [], []
        ensemble = []
        for i, (trainData, testData) in tqdm(enumerate(zip(trainDatas, testDatas)), total=GlobalParams.testFolds,
                                             postfix=f"{species} on {classifier.value}[{defs.getClassifier(classifier,cdf.vars).n_jobs}]"):


            clf.fit(trainData.X, trainData.y)

            #create a copy of clf
            ensemble.append(clf.best_estimator_)

            testScores += [clf.score(testData.X, testData.y)]
            trainScores += [clf.score(trainData.X, trainData.y)]

        #Store the ensemble result, test and training scores are the means of the involved classifiers
        prt = PretrainedVotingClassifier(ensemble)
        model = TrainedModel(species, classifier, cdf, None, scaler, labelEncoder, np.mean(testScores), np.mean(trainScores), prt,None,noiseReduction,NoiseRemovalBalance.Equal)
        model.saveToDisc(not saveData)

        _printSingleResult(testScores, trainScores, model, cdf)

        ret += [model]

        return ret

    except Exception as e:
        termutil.errorPrint(f"            Error for {species} | {classifier} ")
        print(e)


    k = 0


if __name__ == '__main__':
    trainClassifiers(True, [ModelType.GLM], [PredictiveVariableSet.Full], [NoiseRemovalBalance.Equal], [Dataset.AdultsWithSameSplitByDBH], saveModelTrainingData=False)
    pass
