from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

import paths as PATHS
import src.__libs.pyutil as pyutil
from GlobalParams import GlobalParams
from src.__libs import mputil
from src.__libs.pyutil import termutil
from src.classes.Enums import Dataset, PredictiveVariableSet, ClassificationProblem
from src.classes.VariableList import VariableList
from src.datautil import datautil
import os

def _noiseEtaTest():

    #allsame
    allsame = _noiseEta(0,np.array([0,0,0,0]),np.array([1,2,3,6]))
    alldiff = _noiseEta(0,np.array([1,1,1,1]),np.array([1,2,3,4]))
    outlier = _noiseEta(0,np.array([1,0,1,1]),np.array([6,7,8,9]))
    almostOutlierDifType = _noiseEta(0,np.array([1,1,1,1]),np.array([4,6,7,8]))
    almostOutlierSameType = _noiseEta(0,np.array([0,2,2,2]),np.array([4,6,7,8]))
    closeNoise = _noiseEta(0,np.array([1,1,1,1]),np.array([1,1,6,7]))
    closeUnclear = _noiseEta(0,np.array([1,0,1,1]),np.array([1,1,6,7]))

    k = 0
    #


def _noiseEta(pointLbl, lbls:np.ndarray, dists:np.ndarray, maxDist:float = 5):
    """Gives a noise score, where 0 corresponds to not noise and 1 to more likely noise"""

    #transform distance to influence. The further away the less influence
    #Maximum influence is the number of neighbours within the boundary.
    dists = 1 - dists/maxDist
    dists[dists < 0] = 0

    if np.sum(dists) == 0: return 0,0,0 #This point is an outlier and we can't label it as noise

    dists = (np.count_nonzero(dists) / len(dists)) * dists / np.sum(dists)


    lbls = lbls != pointLbl
    lbls = lbls.astype(int)

    return np.sum(lbls * dists), np.count_nonzero(dists) ,np.count_nonzero((1-lbls) * dists)

def computeNoiseScores(overwrite:bool, dataset:Dataset, varset:VariableList):
    """
    Compute the noise scores for the given combination and variable set. The noise score is a measure of how likely a point/site in the training set is to be noise based on its label (increase/decrease in abundance) and the distance to its k nearest neighbours.
    :param overwrite: If true will overwrite existing files
    :param dataset: The dataset to use
    :param varset: The variable set to use
    :return: None, results stored in a CSV file
    """

    """ALGORITHM PARAMS"""
    _k = [GlobalParams.noise_k]
    _alpha = [GlobalParams.noise_alpha]
    _metrics = [GlobalParams.noise_metric]

    #check if the data is already computed
    if os.path.exists(PATHS.Noise.noiseLabels(dataset, varset)) and not overwrite:
        termutil.successPrint("Skipping noise score computation - file already exists")
        return None

    data = datautil.loadTrainingData(dataset, ClassificationProblem.IncDec, varset, False, True, None)

    sgr = data.groupby("Species")

    stds = StandardScaler()


    curR = []
    for species,pnts in tqdm(sgr):
        ids = pnts.loc[:,["PlotID","Year0","Year1","Type"]]
        X = stds.fit_transform(pnts[varset.list])
        y = pnts["Type"]
        le = LabelEncoder()
        y = le.fit_transform(y)


        #find the average distance between points
        for m in _metrics:
            nearest = NearestNeighbors(n_neighbors=max(_k)+1,algorithm="ball_tree",metric=m).fit(X)
            distances,indices = nearest.kneighbors(X)
            avgDist = distances.mean(axis=0) #avg Distance to the kth nearest neighbour. Setting alpha =1 ensures we have on average k neighbours.

            for k in _k:
                kdist = distances[:,1:k+1]
                kind = indices[:,1:k+1]
                for alpha in _alpha:
                    for i in range(len(X)):
                        noise,validNeighbours,samelabelNeighbours = _noiseEta(y[i],y[kind[i]],kdist[i],avgDist[k]*alpha)
                        curR += [list(ids.iloc[i]) + [species,k,alpha,noise,validNeighbours,samelabelNeighbours,m]]

    res = pd.DataFrame(curR,columns=["PlotID","Year0","Year1","Type","Species","k","alpha","Noise","Neighbours","SameLabelNeighbours","Metric"])


    dataCols = ["Noise","Neighbours","SameLabelNeighbours"]
    idCols = list(set(res.columns) - set(dataCols))

    assert len(res) == len(res.drop_duplicates(subset=idCols,keep="last"))

    pyutil.writePandasToCSV(res, PATHS.Noise.noiseLabels(dataset, varset), "Noise Labels", index=False, float_format=GlobalParams.floatFormat)



if __name__ == '__main__':
    _noiseEtaTest()

    args = list(product([Dataset.AdultsOnly], [PredictiveVariableSet.PC7]))
    mputil.runParallel(computeNoiseScores,args,len(args),debug=False)

