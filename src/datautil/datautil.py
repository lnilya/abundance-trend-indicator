import math
from collections import defaultdict
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import paths as PATHS
from GlobalParams import GlobalParams
from src.__libs.pyutil import excludeFromDF, termutil
from src.classes.ClassifierDataSet import ClassifierDataSet
from src.classes.Enums import ClassCombinationMethod, ClassificationProblem, ClassLabels, NoiseRemovalBalance, ModelType
from src.classes.Types import SpeciesName
from src.classes.VariableList import VariableList


def getOccurrenceData(addProperties:List = [], usecols = None):
    """Loads the occurrence dataset and if needed merges it with the plotInfo to have one dataframe describing occurrence and plot information."""
    plotInfo = getAllPlotInfo(True, addProperties)
    allRanges = pd.read_csv(PATHS.Occ.Combined, usecols=usecols)
    #add the plot info without adding any new columns
    allRanges = allRanges.merge(plotInfo,on=["PlotID","ObservationID"],how="left")
    return allRanges,plotInfo

def loadMigrationDF(staticProps:Optional[VariableList], comb:ClassCombinationMethod, climProps:VariableList = None, problem:ClassificationProblem = ClassificationProblem.IncDec):
    """Loads the occurrence dataset and if needed merges it with the plotInfo to have one dataframe describing occurrence and plot information."""

    baseCols = ['PlotID', 'Species', 'Year0', 'Year1', 'Type', 'TypeNumeric']
    if climProps is not None:
        baseCols += climProps.list
        baseCols = list(set(baseCols)) #Remove duplicates
    else:
        baseCols = None #Load all columns if none

    allMigrations = pd.read_csv(PATHS.Shifts.allPlotsCombined(comb,True), usecols=baseCols)

    # Discard the types we are not interested in (especially uncertain type)
    allMigrations = allMigrations.loc[allMigrations.Type.isin(problem.toLabels(str)), :]

    plotInfo = None
    if staticProps is not None:
        spl = list(set(staticProps.list) - set(allMigrations.columns))
        if len(spl) > 0:
            plotInfo = getAllPlotInfo(False, staticProps.list)
            allMigrations = allMigrations.merge(plotInfo,on=["PlotID"],how="left")

    return allMigrations,plotInfo


def getAllPlotInfo(preserveObservationIDs:bool = True, geoProperties:Optional[List] = None, mergeWith:pd.DataFrame = None):
    """Returns a dataframe with all plot information
    :param preserveObservationIDs: If True, the dataframe will contain the original observation IDs. This means some parentIDs will have multiple entries.
    If False will also remove the Year column
    :param geoProperties: Loads the properties file and adds the static properties from NZENVDS layers.
    """
    if geoProperties is None:
        plotCoords = pd.read_csv(PATHS.PlotInfo.WithGeoProps)
    else:
        plotCoords = pd.read_csv(PATHS.PlotInfo.WithGeoProps, usecols=["PlotID","ObservationID"] + geoProperties)


    if not preserveObservationIDs:
        if "Year" in plotCoords.columns:
            plotCoords.drop(columns=["Year"], inplace=True)
        plotCoords = plotCoords.groupby("PlotID").first().reset_index()
        plotCoords.drop(columns=["ObservationID"], inplace=True)

    if mergeWith is not None:
        mergeCols = ["PlotID"]
        if preserveObservationIDs and "ObservationID" in mergeWith.columns:
            mergeCols.append("ObservationID")
        mergeWith = mergeWith.merge(plotCoords, how="left", on=mergeCols)

        return mergeWith,plotCoords

    return plotCoords


def _excludeSpeciesWithTooFewEntries(data,silent = False)->pd.DataFrame:

    #Exclude species with too few observations per class
    countsBySpeciesAndType = data.groupby(["Species","Type"]).count().loc[:,"PlotID"].reset_index()
    excludedByClass = set(countsBySpeciesAndType[countsBySpeciesAndType.PlotID < GlobalParams.minObservationsPerClass].Species)
    if not silent:
        print(f"    Excluded {len(excludedByClass)} species because observations < {GlobalParams.minObservationsPerClass}")

    #Exclude species with too few observations total
    countsBySpecies = data.groupby(["Species"]).count().loc[:,"PlotID"].reset_index()
    excludedByTotal = set(countsBySpecies[countsBySpecies.PlotID < GlobalParams.minObservationsTotal].Species)

    if not silent:
        print(f"    Excluded additional {len(excludedByTotal-excludedByClass)} species because total observations < {GlobalParams.minObservationsTotal}")


    return data.loc[~data.Species.isin(excludedByClass)&~data.Species.isin(excludedByTotal)]


def loadTrainingData(comb:ClassCombinationMethod, classificationProblem:ClassificationProblem, allVars:VariableList, addRandomColumn:bool = False, returnAsDF:bool = False, removeNoise:float = None, silent = False, nrb:NoiseRemovalBalance = NoiseRemovalBalance.Equal)->Union[pd.DataFrame,dict[SpeciesName, ClassifierDataSet]]:
    """Loads the data in the correct format for different classification problems and variables. Ready to be used by classifiers
    :param classificationProblem: The classification problem to solve @see ClassificationType
    :param allVars: The variables to use @see VariableList
    :param addRandomColumn: If true adds a random column to the data, useful to check if the classifier is overfitting
    :param returnAsDF: If True, returns the data as a dataframe, otherwise as a dictionary with the species as keys
    :param removeNoise: If not None, removes the top removeNoise percentage of the data with the highest noise scores
    :param silent: If True, does not print any output
    :param nrb: The way noise removal between different classes is handled
    :return: A dictionary with the species as keys and the data as values
    """

    #TODO: Remove the classificationProblem parameter

    data = pd.read_csv(PATHS.Shifts.allPlotsCombined(comb), usecols=["Species","Type","Year0","Year1","PlotID"] + allVars.list)
    data[allVars.list] = data[allVars.list].apply(pd.to_numeric, downcast="float", errors="coerce")
    #count species
    data.groupby("Species").size().reset_index(name="Count").sort_values("Count",ascending=False)


    #Filter out observations that are too far apart
    data["dY"] = data["Year1"] - data["Year0"]
    data = excludeFromDF(data,data.dY <= GlobalParams.maxYearDifference,"Year difference too large",silent)

    if addRandomColumn:
        allVars = allVars + ["Random"]
        data["Random"] = np.random.rand(len(data))

    #Discard the types we are not interested in (especially uncertain type)
    data = data.loc[data.Type.isin(classificationProblem.toLabels(str)),:]

    # Convert to 2-class problem by collapsing the other two into the rest class.
    if classificationProblem == ClassificationProblem.IncRest:
        mask = data.Type.isin([ClassLabels.Same, ClassLabels.Dec])

        data.loc[mask, "TypeNumeric"] = 0
        data.loc[mask, "Type"]= ClassLabels.DecSame
    elif classificationProblem == ClassificationProblem.DecRest:
        mask = data.Type.isin([ClassLabels.Same, ClassLabels.Inc])
        data.loc[mask, "TypeNumeric"] = 0
        data.loc[mask, "Type"] = ClassLabels.IncSame

    data = _excludeSpeciesWithTooFewEntries(data,silent)

    if removeNoise is not None and removeNoise > 0:
        noise = pd.read_csv(PATHS.Noise.noiseLabels(comb,allVars), usecols=["PlotID","Year0","Year1","Species","Noise","k","alpha","Metric"])
        noise = noise.loc[(noise.Metric == GlobalParams.noise_metric) & (noise.k == GlobalParams.noise_k)&(noise.alpha == GlobalParams.noise_alpha),:]
        noise.drop(columns = ["k","alpha","Metric"],inplace=True)
        noise.Noise = noise.Noise.astype(np.float32)
        #find duplicate row ids
        oldLen = len(noise)
        noise = noise.drop_duplicates(subset=["PlotID","Year0","Year1","Species"])
        if len(noise) != oldLen and not silent:
            termutil.errorPrint("Noise dataframe contains duplicates. they are removed. This should not happen.")


        #add the noise to the data
        data = data.merge(noise, on=["PlotID","Year0","Year1","Species"], how="left")

        #ensure there are no blanks
        assert data.Noise.notna().all()


        def _remNoiseMax(x):
            x.sort_values("Noise",ascending=False,inplace=True)
            #retain only the 1-removeNoise percentage of the data
            return x.iloc[int(len(x)*removeNoise):,:]

        def _remNoiseProp(group):
            grI = group[group.Type == "I"].sort_values("Noise",ascending=True)
            grD = group[group.Type == "D"].sort_values("Noise",ascending=True)

            classProportions = {k: v / len(group) for k, v in group.Type.value_counts().to_dict().items()}

            rD = removeNoise / (1 + (len(grI) / len(grD)))
            rI = removeNoise - rD
            sD = round((1-rD) * len(grD))
            sI = round((1-rI) * len(grI))

            grIClean = grI.iloc[:sI, :]
            grDClean = grD.iloc[:sD, :]
            return pd.concat([grIClean, grDClean])
            #retain only the 1-removeNoise percentage of the data
        def _remNoiseEqual(group):
            grI = group[group.Type == "I"].sort_values("Noise",ascending=True)
            grD = group[group.Type == "D"].sort_values("Noise",ascending=True)

            sI = round(len(grI) * (1-removeNoise))
            sD = round(len(grD) * (1-removeNoise))

            grIClean = grI.iloc[:sI, :]
            grDClean = grD.iloc[:sD, :]
            return pd.concat([grIClean, grDClean])
            #retain only the 1-removeNoise percentage of the data



        if nrb == NoiseRemovalBalance.Combined:
            data = data.groupby("Species").apply(_remNoiseMax).reset_index(drop=True)
        elif nrb == NoiseRemovalBalance.Equal:
            data = data.groupby("Species").apply(_remNoiseEqual).reset_index(drop=True)
        elif nrb == NoiseRemovalBalance.Proportional:
            data = data.groupby("Species").apply(_remNoiseProp).reset_index(drop=True)
        else:
            raise ValueError("Noise removal balance is not specified/not implemented."
                             "")
        #exclude more species if the noise cleaning led them to fall below the threshold.
        if not silent:
            print(f"    Excluded {removeNoise*100:.2f}% of data with highest noise scores. Rerunning minimal data checks:")
        data = _excludeSpeciesWithTooFewEntries(data,silent)



    if returnAsDF:
        if not silent:
            termutil.infoPrint("    Loaded Training data with " + str(data.Species.nunique()) + " species.")
            print("")
        return data

    sp = data.groupby("Species")
    res:Dict[SpeciesName, ClassifierDataSet] = {}
    for s,spGroup in sp:

        X = spGroup.loc[:, allVars]

        uniqueX = np.unique(X, axis=0)

        ratio = 100 - 100 * len(uniqueX) / len(X)
        if ratio > 2 and not silent:
            termutil.errorPrint(f"    Some data is duplicated for species {s}. If the rate is small, it is likely ok. {ratio:.2f}%")

        plotKeys = spGroup.loc[:,["PlotID","Year0","Year1"]]
        #cast all to ints
        plotKeys = plotKeys.astype(int)
        res[s] = ClassifierDataSet(removeNoise,comb,classificationProblem, allVars, X.to_numpy(), spGroup.loc[:, "Type"].to_numpy(),plotKeys)

    #sort dictionary by Number observations
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1].numObs, reverse=True)}

    if not silent:
        termutil.infoPrint("    Loaded Training data with " + str(data.Species.nunique()) + " species.")
        print("")
    return res
