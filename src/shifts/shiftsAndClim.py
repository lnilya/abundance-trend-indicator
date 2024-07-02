import pandas as pd
import numpy as np
from tqdm import tqdm

import paths as PATHS
import src.__libs.pyutil as pyutil
from GlobalParams import GlobalParams
from src.__libs import mlutil
from src.__libs.pyutil import termutil
from src.classes.Enums import Dataset, PredictiveVariableSet
from src.classes.VariableList import VariableList
from src.datautil import getAllPlotInfo
import os

def createTrainingData(overwrite:bool, dataset: Dataset):
    """
    Transforms the raw data into the training data by combining the target labels with the predictor variables.
    :param overwrite: If true will overwrite existing files
    :param dataset: The dataset to use
    :return: None, results stored in a CSV file
    """
    if os.path.exists(PATHS.Shifts.allPlotsCombined(dataset)) and not overwrite:
        termutil.successPrint("Skipped Training Data Creation. File exists.")
        return

    termutil.chapPrint(f"Building training data for {dataset.name}")


    climatePlots = pd.read_csv(PATHS.Bioclim.AtPlotsByYearLinearAppx)
    shifts = pd.read_csv(PATHS.Shifts.allPlotsByMethod)


    if dataset == Dataset.AdultsWithSameSplitByDBH:
        # Adult contains the class by number of individuals
        # AdultTA contains the class by DBH
        # Where Adult is S we use AdultTA

        shifts.rename(columns={"Adult": "Type"}, inplace=True)
        # drop the nans
        shifts = shifts.dropna(subset=["Type"]).reset_index(drop=True)

        #Compute proportion of I or D in AdultTA when Type is S
        sameIdx = np.where(shifts.Type == "S")
        dd = shifts.loc[sameIdx, "AdultTA"].value_counts().to_dict()
        dds = len(shifts.loc[sameIdx, "AdultTA"])
        percentageIncThatIsSurvival = 100 * dd["I"] / dds
        percentageDecThatIsSurvival = 100 * dd["D"] / dds
        percentageSameThatIsSurvival = 100 * dd["S"] / dds
        print(f"When number of individual stays the same {percentageIncThatIsSurvival:.2f} % are increases. Which can be counted as survival. So Inc class will now contain survival. ")
        print(f"When number of individual decreases {percentageDecThatIsSurvival:.2f} % are increases. Which can be counted as survival. So Dec class will now contain survival. ")
        print(f"When number of individual stays the same {percentageSameThatIsSurvival:.2f} % are increases. Which can be counted as survival. So Same class will now contain survival. ")

        # Replace count values with dbh where Adult is S
        shifts.Type = np.where(shifts.Type == "S", shifts.AdultTA, shifts.Type)

    elif dataset == Dataset.AdultsOnly:
        # Average the shifts
        shifts.rename(columns={"Adult": "Type"}, inplace=True)
        # drop the nans
        shifts = shifts.dropna(subset=["Type"]).reset_index(drop=True)

    # Replace "A" with "I" and "R" with "D"
    shifts.replace({"Type": {"A": "I", "R": "D"}}, inplace=True)

    columns = [f"BIOEnd{i}" for i in range(1, 20)]  # Value at last Year

    # Add the climate data
    allTuples = shifts.groupby(["PlotID", "Year0", "Year1"])
    for (pid, y0, y1), gr in tqdm(allTuples):
        r = climatePlots[(climatePlots.PlotID == pid) & ((climatePlots.Year == y0) | (climatePlots.Year == y1))]
        r = r.loc[:, ["Year"] + columns].sort_values("Year")
        #sort the columns of r so they are in the same order as columns

        bioEnd = r.iloc[1, 1:].to_list()

        # add to the shifts as columns to the dataframe

        shifts.loc[gr.index, columns] = bioEnd

    _addGeoVariablesToTrainingData(shifts, dataset)

    pass


def _addGeoVariablesToTrainingData(data: pd.DataFrame = None, cc: Dataset = Dataset.AdultsWithSameSplitByDBH):

    if data is None:
        data = pd.read_csv(PATHS.Shifts.allPlotsCombined(cc))

    originalLength = len(data)


    geoColumns = list(set(PredictiveVariableSet.Full).union(["mapX", "mapY"]) - set(data.columns))

    if len(geoColumns) > 0:
        plotInfo = getAllPlotInfo(False, geoColumns)
        # merge the plotInfo to the data
        data = data.merge(plotInfo, on=["PlotID"], how="left")

    assert not np.any(data.loc[:,PredictiveVariableSet.Full.list].isna()), "Some entries in the data are empty. Check the data."

    # Compute the PCs
    # collapse the datapoints
    dataPoints = data.groupby(["PlotID", "Year0", "Year1"]).first().reset_index()

    assert len(data) == originalLength, "Dataframe length changed. Check the data."

# add the data back

    pyutil.writePandasToCSV(data, PATHS.Shifts.allPlotsCombined(cc), "Done building training data",
                        index=False, float_format=GlobalParams.floatFormat)

if __name__ == '__main__':
    # addGeoVariablesToDataframe(cc = ClassCombinationMethod.AdultsWithSameSplitByDBH)
    createTrainingData(Dataset.AdultsWithSameSplitByDBH, True)
    # condenseShiftsAndAddClim(ClassCombinationMethod.AdultsOnlyByArea,True)
    # condenseShiftsAndAddClim(ClassCombinationMethod.AdultsOnly,True)
    # condenseShiftsAndAddClim(ClassCombinationMethod.SaplingsOnly,False)
    # condenseShiftsAndAddClim(ClassCombinationMethod.CoverOnly,False)
