from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import paths as PATHS
from GlobalParams import GlobalParams
from src.__libs import mputil
from src.classes.Enums import ClassCombinationMethod, ClassificationProblem, PredictiveVariableSet
from src.classes.FileIDClasses import SimilarityDataFileID
from src.classes.SimilarityData import SimilarityData
from src.classes.StackedData import StackedData
from src.classes.VariableList import VariableList
from src.datautil import datautil


def computeSimilarity(overwrite, dataset: ClassCombinationMethod, varset: VariableList = PredictiveVariableSet.Full):
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

    occData = pd.read_csv(PATHS.Occ.Combined, usecols=["ParentPlotID", "Species"])
    occData, _ = datautil.getAllPlotInfo(False, ["mapX", "mapY"], occData)

    for species, spGroup in spg:
        # compute the similarity between the training points and the data
        # using nearest neighbour distance
        simid = SimilarityDataFileID(yearRange, GlobalParams.similarity_k, GlobalParams.similarity_metric, str(species), varset, dataset, ClassificationProblem.IncDec)
        if simid.fileExists() and not overwrite:
            print(f"Skipped {species}. Already exists.")
            continue
        allDistances = []
        for yr in tqdm(yearRange):

            d = data[yr]
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
            allDistances.append(np.mean(distances,axis=1))

        meanDistances = np.mean(allDistances, axis=0)

        sd = SimilarityData(simid,meanDistances,data[yearRange[0]].nanmask,data[yearRange[0]].shape)
        sd.computeNormalization(occData)
        sd.saveToDisc()



if __name__ == '__main__':
    # computeSimilarity(5,"manhattan",ClassCombinationMethod.AdultsWithSameSplitByDBH)
    args = list(product([1,3,5],["l1","l2"],[ClassCombinationMethod.AdultsWithSameSplitByDBH]))
    mputil.runParallel(computeSimilarity, args, len(args))

    pass
