import os
from typing import List, Any

import pandas as pd
from tqdm import tqdm

import paths as PATHS
import src.__libs.pyutil as pyutil
from GlobalParams import GlobalParams
from src.__libs.pyutil import termutil
from src.classes.Enums import ClassLabels
from src.datautil import datautil


def _printDebug(yg0,yg1,col):
    y0 = yg0.Year.iloc[0]
    y1 = yg1.Year.iloc[0]
    s0 = set(yg0[yg0[col] > 0].index)
    s1 = set(yg1[yg1[col] > 0].index)
    print(f"[{col}] Present Species in {y0}: ", yg0.loc[yg0[col] > 0, col])
    print("")
    print(f"[{col}] Present Species in {y1}: ", yg1.loc[yg1[col] > 0, col])
    print("")
    overlap = s0.intersection(s1)
    print(f"[{col}] Inc/Dec/Same Species: ", overlap)

    sAdd = s1 - s0
    sRem = s0 - s1
    print("Added species ", sAdd)
    print("")
    print("Removed species ", sRem)
    print("")
    k = 0

def _getClassification(yg0, yg1, mainQuantityColName:str, classColName:str, *additionalQuantityColumns)->pd.DataFrame:
    """
    Classifies the difference between yg0 and yg1 for a given column.
    See unit test at the bottom of file for more info.

    # NaN -> +   This species just migrated and will be considered "ADD"
    # + -> NaN   This species just disappeared and will be considered "REMOVE"
    # 0/+ -> ++   This species increased and will be considered "INCREASE"
    # ++ -> 0/+   This species decreased and will be considered "DECREASE"
    # + -> +  This species stayed the same and will be considered "STAY"
    # 0 -> 0 This species will be ignored as it is the same as NaN->NaN it wasnt present at all


    :param yg0: Species counts for year 0
    :param yg1: Species counts for year 1
    :param mainQuantityColName: The column where the counts are stored
    :return: a Series where the index is the species and the values are the classification of the difference
    """

    dif = yg1[mainQuantityColName] - yg0[mainQuantityColName]
    #We filter out all elements that are 0 in both years, as they are not interesting for us.
    #These will end up in the "Same" category otherwise, but we do not want this.
    difc = dif.copy().astype(str)

    difc[dif > 0] = ClassLabels.Inc.value
    difc[dif < 0] = ClassLabels.Dec.value
    difc[(dif == 0)] = ClassLabels.Same.value

    # dif will be NaN for species mentioned only in one of the years. We can use this to determine ADD/REMOVE
    removed = dif.index[dif.index.isin(yg0[yg0[mainQuantityColName] > 0].index) & dif.isna()]
    added = dif.index[dif.index.isin(yg1[yg1[mainQuantityColName] > 0].index) & dif.isna()]
    difc.loc[removed] = ClassLabels.Rem.value
    difc.loc[added] = ClassLabels.Add.value

    difc = difc.loc[(difc != "nan") & ~((yg0[mainQuantityColName] == 0) & (yg1[mainQuantityColName] == 0))]
    difc = pd.DataFrame(difc)
    difc.rename(columns={mainQuantityColName: classColName}, inplace=True)
    difc[mainQuantityColName] = dif[difc.index]

    if len(additionalQuantityColumns) > 0:
        difAdd = yg1.loc[:,additionalQuantityColumns] - yg0.loc[:,additionalQuantityColumns]
        difc = difc.join(difAdd, how="left")

    additionalQuantityColumns = [mainQuantityColName] + list(additionalQuantityColumns)
    #Added/Removed naturally have NaNs in the count column, so we set them to the actual value by which it decreased/increased
    difc.loc[added,additionalQuantityColumns] = yg1.loc[added,additionalQuantityColumns]
    difc.loc[removed,additionalQuantityColumns] = -yg0.loc[removed,additionalQuantityColumns]

    return difc

def identifyRemeasuredPlots(overwrite:bool = False):
    """
    Identifies plots that have been remeasured in the raw data and stores the results into a CSV file, this does not yet create the target labels, but is a precursor to the training data.
    :param overwrite: If true will overwrite existing files
    :return: None, results stored in a CSV file
    """

    if os.path.exists(PATHS.Shifts.allPlotsByMethod) and not overwrite:
        termutil.successPrint("Skipped Migration Analysis. File exists.")
        return

    termutil.chapPrint("Migration plot analysis")
    pyutil.tic()

    # data, plotinfo = datautil.getOccurrenceData(["Year"] + plotInfoCols,["ParentPlotID","Species","ParentPlotObsID"])
    data, plotinfo = datautil.getOccurrenceData(["Year"])

    # Get observation pairs and store into a new dataframe
    allPlots = data.groupby("PlotID")


    results = []
    countsByRemeasurements = dict()
    for plotID, plotGroup in tqdm(allPlots):
        # Get all Observations sort by year and ensure that species are shared in both batches, imputing 0 where needed
        years = plotGroup.Year.unique()
        if len(years) <= 1: continue
        years.sort()
        yearTuples = list(zip(years[:-1], years[1:]))

        countsByRemeasurements[len(yearTuples)] = countsByRemeasurements.get(len(yearTuples), 0) + 1

        for (y0, y1) in yearTuples:
            if y0 >= GlobalParams.maxYear or y1 >= GlobalParams.maxYear: continue
            if y0 < GlobalParams.minYear or y1 < GlobalParams.minYear: continue

            yg0 = plotGroup[plotGroup.Year == y0].set_index("Species")
            yg1 = plotGroup[plotGroup.Year == y1].set_index("Species")

            allDifs:List[Any] = []
            #check if both columns contain non-NaN elements, this means that the method was applied to both observations
            if yg0["NumIndividuals"].notna().any() and yg1["NumIndividuals"].notna().all():
                difc = _getClassification(yg0,yg1,"NumIndividuals","Adult","Total Diameter")
                allDifs.append(difc)

            #This happens if the species was measured with different methods at the different time-spots and there is no overlap
            #These species could all be added as "Uncertain" but we just don't care as uncertain is not used anywhere.
            if len(allDifs) == 0: continue

            #Merge allDifs into a dataframe
            df = pd.concat(allDifs, axis=1)
            df = df.assign(Year0=y0, Year1=y1, PlotID=plotID)
            df.reset_index(inplace=True)
            results += [df]


    # Convert counts to dataframe
    # Add the additional columns straight to the data
    plotinfo.drop(columns=["Year"], inplace=True)
    plotinfo.groupby("PlotID").first().reset_index(inplace=True)
    inv = pd.concat(results, axis=0,ignore_index=True)

    pyutil.writePandasToCSV(inv, PATHS.Shifts.allPlotsByMethod, "Migration Plots (Decrease/Increase)", index=False, printFun=termutil.endPrint, float_format=GlobalParams.floatFormat)
    # inv,vars,loadings = postProcessDataFrame(resDecIn, columns, plotinfo, plotInfoUnchangedCols)
    # pyutil.writeExcelWithSheets(PATHS.allMigrationPCALoadings % by, {"Loadings":loadings, "Variances":vars})


    pyutil.toc("Migration Analysis")

    pass

def _runUnitTest():
    # NaN means that no datasets (seedlings,saplings,cover,stems) have recorded the species
    # 0 means that the species was recorded by at least one of the datasets but no individuals were found in this particular datasets.
    # e.g. if seedlings of species X are present but no adults, the adult entry will be 0 for X. Species Z is not present as seedling nor step, so this entry would have a NaN value.
    # These NaN values only happen once you compare the two years, since both datasets have >= 0 entries at all times. This happens because species set in year0 is not identical to year1.

    # Controversy: A 0 means essentially that the species is NOT PRESENT according to a the dataset. so going from 0 -> + can be considered an ADD not an INC.
    # However since the species seems to be present in at least some other dataset, it is considered an INC. This maakes sense when the other dataset are seedlings and saplings, and the one we look
    # at are individuals. If individuals goes from 0 -> + it means that the species has grown up and is now present in the adult dataset, so it should be considered an INC rather than an ADD.

    # For classification it may not make any difference as we will likely only look ad a combined ADD+INC vs DEC+REM classification.

    # NaN -> +   This species just migrated and will be considered "ADD"
    # + -> NaN   This species just disappeared and will be considered "REMOVE"
    # 0/+ -> ++   This species increased and will be considered "INCREASE"
    # ++ -> 0/+   This species decreased and will be considered "DECREASE"
    # + -> +  This species stayed the same and will be considered "STAY"
    # 0 -> 0 This species will be ignored as it is the same as NaN->NaN it wasnt present at all

    testY0 = pd.DataFrame({"species": ["Same(S)", "Same0(Should not occur)", "AddFrom0(I)", "Remove(D)", "RemovedToNaN(R)", "Inc(I)", "Dec(D)","Unique0(Should not occur)"],
                           "count": [1, 0, 0, 3, 3, 4, 5, 0],
                           "difcol": [1,2,3,4,5,6,7,8],
                           "difcol2": [8,7,6,5,4,3,2,1]
                           })
    testY1 = pd.DataFrame({"species": ["Same(S)", "Same0(Should not occur)", "AddFrom0(I)", "AddFromNaN(A)", "Remove(D)", "Inc(I)", "Dec(D)","Unique1(Should not occur)"],
                           "count": [1, 0, 2, 3, 0, 5, 4, 0],
                           "difcol": [8,7,6,5,4,3,2,1],
                           "difcol2": [1,2,3,4,5,6,7,8]
                           })

    testY1.set_index("species", inplace=True)
    testY0.set_index("species", inplace=True)

    r = _getClassification(testY0, testY1, "count","labelcol", "difcol", "difcol2")
    return r


if __name__ == '__main__':
    r = _runUnitTest()
    print(r)
    identifyRemeasuredPlots()
