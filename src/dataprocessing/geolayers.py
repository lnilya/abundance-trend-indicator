import multiprocessing
import os

import pandas as pd
import rasterio
from tqdm import tqdm

import paths as PATHS
# import src.__libs.gis as gisutil
import src.__libs.pyutil as pyutil
from GlobalParams import GlobalParams
from src.__libs import mputil
from src.__libs.mlutil import mlutil
from src.__libs.osutil import getAllFiles
from src.__libs.pyutil import termutil
import numpy as np



#Function for multiprocess run that catches errors
def _extractSafe(tifFilePath, coords, primaryIDs, colName, msgAdd:str = "", silent = False):

    if not silent:
        print("     Extracting %s %s"%(colName,msgAdd))
    try:
        with rasterio.open(tifFilePath) as src:
            rawData = src.read(1)
            res:np.ndarray = rawData[coords.mapY, coords.mapX]
            #assert none are nans
            assert not np.isnan(res).any(), "Some plots are on NaN values in the geotiff"

            return pd.DataFrame({"PlotID":primaryIDs, colName:res})

    except:
        termutil.errorPrint("Failed processing %s"%tifFilePath)
        return None


def extractClimateLayersLinAppx(overwrite:bool = False):
    """
    Takes the climate values for each plot and creates a linear regression for each plot, the values of this linear regressions are stored
    this eliminates the noise and focuses on the large-scale changes in climate.
    This means that the output
    :return:
    """
    if os.path.exists(PATHS.Bioclim.AtPlotsByYearLinearAppx) and not overwrite:
        termutil.successPrint("Skipped Climate Layer Linear Approximation. File exists.")
        return

    termutil.chapPrint("Creating linear approximations of climate data by year")
    pyutil.tic()
    rng = GlobalParams.yearRange

    climateAllYears = pd.read_csv(PATHS.Bioclim.AtPlotsByYear)
    climateAllYears = climateAllYears[(climateAllYears.Year <= rng[1])&(climateAllYears.Year >= rng[0])]
    # hotRunzAllYears = hotRunzAllYears[hotRunzAllYears.ParentPlotID < 200]
    gr = climateAllYears.groupby("PlotID")
    for pid,group in tqdm(gr):
        #make lin apprx for each column and replace the values with the lin apprx
        for col in group.columns:
            if col in ["PlotID","Year"]: continue
            slope,intercept = np.polyfit(group.Year,group[col],1)
            climateAllYears.loc[group.index,col] = slope * group.Year + intercept

    pyutil.writePandasToCSV(climateAllYears,PATHS.Bioclim.AtPlotsByYearLinearAppx,printFun=termutil.successPrint,index=False, float_format=GlobalParams.floatFormat)
    pyutil.toc("Linear Approximation")
    pass

def extractClimateLayers(overwrite:bool = False):

    if os.path.exists(PATHS.Bioclim.AtPlotsByYear) and not overwrite:
        termutil.successPrint("Skipped Climate Layers. File exists.")
        return

    termutil.chapPrint("Extracting year dependent climate data from geotiff layers")
    pyutil.tic()

    # Load PlotData

    plotDataWithObs = pd.read_csv(PATHS.PlotInfo.NoProps)
    plotData = plotDataWithObs.groupby("PlotID").first().reset_index()
    coords = plotData.loc[:, ["mapX", "mapY"]]

    #get all folders
    allFolders = [f for f in os.listdir(PATHS.Raw.Predictors) if os.path.isdir(PATHS.Raw.Predictors+f)]

    mergedDF = None
    for name in tqdm(allFolders,desc="Climate Data"):
        yearDF = []
        for y in range(GlobalParams.minYear,GlobalParams.maxYear):
            df = _extractSafe(f"{PATHS.Raw.Predictors}{name}/{y}.tif", coords, plotData.PlotID, name, str(y), True)
            df["Year"] = y
            yearDF += [df]

        #merge this year into a single DF
        yearDF = pd.concat(yearDF)

        #add this to the main DF
        if mergedDF is None: mergedDF = yearDF
        else: mergedDF = mergedDF.merge(yearDF, on=["PlotID","Year"], how="left")

    pyutil.writePandasToCSV(mergedDF, PATHS.Bioclim.AtPlotsByYear, index=False,
                            float_format=GlobalParams.floatFormat, printFun=termutil.endPrint)
    pyutil.toc("Extracting", printFun=termutil.endPrint)



def extractGeoLayers(overwrite:bool = False):
    """
    Loads the geotiff files provided by NZENVDS and reads their values at the locations of the measured plots.
    The results are then added to the same file as the plot info.
    :return: The dataframe with the results (but DF will also be saved to file)
    """
    if os.path.exists(PATHS.PlotInfo.WithGeoProps) and not overwrite:
        termutil.successPrint("Skipped Geo Layers. File exists.")
        return
    termutil.chapPrint("Extracting year indepent data from geotiff layers")
    pyutil.tic()

    # Load PlotData

    plotDataWithObs = pd.read_csv(PATHS.PlotInfo.NoProps)
    plotData = plotDataWithObs.groupby("PlotID").first().reset_index()
    coords = plotData.loc[:,["mapX","mapY"]]

    allVars = getAllFiles(PATHS.Raw.Predictors, "*.tif")
    results = []
    for path,name in allVars.items():
        results.append(_extractSafe(path,coords,plotData.PlotID,name))

    #Results is a list of dataframes (or none if failed), merge them together
    for res in results:
        if res is None: continue
        plotDataWithObs = plotDataWithObs.merge(res, on="PlotID", how="left")

    pyutil.writePandasToCSV(plotDataWithObs, PATHS.PlotInfo.WithGeoProps, index=False, float_format=GlobalParams.floatFormat, printFun=termutil.endPrint)
    pyutil.toc("Extracting",printFun=termutil.endPrint)

def checkForInvalidNZENVDSValues():
    #Check if there are values that are similar to -339999995214436387128800550842202587136.00
    #Which seems to be prpoduced when the coordinates are out of bounds
    df = pd.read_csv(PATHS.PlotInfo.WithGeoProps)

    # Remove all positions where values are invalid. These are spots where the plot is outside the NZENVDS bounds. (Chatham islands mostly)
    inv = -339999995214436387128800550842202587136  # can be any column, sometimes not all columns are invalid
    # get all rows that contain the inv value
    sel = df[df == inv].any(axis=1)
    # remove them
    res = pyutil.excludeFromDF(df, ~sel, "Outside NZENVDS Layer (or undefined)")

    return res

if __name__ == '__main__':

    extractGeoLayers()
    extractEcosystems()

