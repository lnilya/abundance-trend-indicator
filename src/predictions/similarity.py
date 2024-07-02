from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import paths as PATHS
from GlobalParams import GlobalParams
from src.__libs import mputil
from src.__libs.pyutil import termutil
from src.classes.Enums import Dataset, ClassificationProblem, PredictiveVariableSet
from src.classes.FileIDClasses import SimilarityDataFileID
from src.classes.SimilarityData import SimilarityData
from src.classes.StackedData import StackedData
from src.classes.VariableList import VariableList
from src.datautil import datautil


def _computeSingleSpecies(d,varset:VariableList,spGroup:pd.DataFrame):
    X = d.getDataAs1DArray(None, varset)
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # Scale the training data as well
    X_train = ss.transform(spGroup[varset].values)

    # compute the distance
    neigh = NearestNeighbors(n_neighbors=GlobalParams.similarity_k, metric=GlobalParams.similarity_metric)
    neigh.fit(X_train)
    # compute the distance

    distances, indices = neigh.kneighbors(X)
    return np.mean(distances, axis=1)

def computeSimilarity(overwrite, dataset: Dataset, varset: VariableList = PredictiveVariableSet.Full, speciesSubset:list = None):
    """
    Computes the similarity of the training data to the entire prediction area. This is done by computing the distance in the feature space between the pixel where prediction happens and the nearest training point.
    The similarities across all years are then averaged to get a single similarity value for each pixel over the entre prediction period.
    Results are saved to disk.
    :param overwrite: If true will overwrite existing files (else will skip this function if the file already exists)
    :param dataset: The dataset to use, this will determine the training data to use
    :param varset: The variable set to use, this will determine the feature space to use
    :param speciesSubset: If not None, only compute similarity for the species in this list
    :return: None
    """
    termutil.chapPrint("Computing Similarity of Environment to Training Data")
    # load training points
    trData = datautil.loadTrainingData(dataset, ClassificationProblem.IncDec, varset, False, True, silent=True)

    spg = trData.groupby("Species")

    yearRange = list(range(GlobalParams.yearRange[0], GlobalParams.yearRange[1]))

    print(f"Running Similarity computation for ({len(spg)} Species):")
    print(f"Vars: {varset.name}")
    print("")

    # load the data itself
    data = {}
    for yr in yearRange:
        data[yr] = StackedData.readFromDisc(yr, PredictiveVariableSet.Full)

    occData = pd.read_csv(PATHS.Occ.Combined, usecols=["PlotID", "Species"])
    occData, _ = datautil.getAllPlotInfo(False, ["mapX", "mapY"], occData)

    for species, spGroup in spg:
        if speciesSubset is not None and species not in speciesSubset:
            continue
        # compute the similarity between the training points and the data
        # using nearest neighbour distance
        simid = SimilarityDataFileID(yearRange, GlobalParams.similarity_k, GlobalParams.similarity_metric, str(species), varset, ClassificationProblem.IncDec,dataset)
        if simid.fileExists() and not overwrite:
            print(f"Skipped {species}. Already exists.")
            continue
        allDistances = []
        args = []
        for yr in yearRange:
            args.append((data[yr],varset,spGroup))

        allDistances = mputil.runParallel(_computeSingleSpecies, args, poolSize=GlobalParams.parallelProcesses, progressMessage=f"{species}")

        meanDistances = np.mean(allDistances, axis=0).reshape(-1,1)

        sd = SimilarityData(simid,meanDistances,data[yearRange[0]].nanmask,data[yearRange[0]].shape)
        sd.computeNormalization(occData)
        sd.saveToDisc()



if __name__ == '__main__':
    # computeSimilarity(5,"manhattan",ClassCombinationMethod.AdultsWithSameSplitByDBH)
    args = list(product([1,3,5], ["l1","l2"], [Dataset.AdultsWithSameSplitByDBH]))
    mputil.runParallel(computeSimilarity, args, len(args))

    pass
