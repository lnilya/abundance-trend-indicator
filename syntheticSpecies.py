import pickle
from random import randint

import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import plotly.graph_objects as go
from tqdm import tqdm

import paths as PATHS
from GlobalParams import GlobalParams
from src.__libs import pyutil
from src.__libs.plotutil import PlotlyGraph, saveAsPrint
from src.__libs.plotutil.PlotlyGraph import PlotlyGraph
from src.__libs.pyutil import termutil
from src.dataprocessing.geolayers import extractGeoLayers, extractClimateLayers, extractClimateLayersLinAppx
import os
import scipy.stats as stats

def _writeSpeciesOccurenceToCSV(df:pd.DataFrame, speciesName:str, append:bool = True):
    colsToKeep = ["ObservationID","PlotID","NumIndividuals"]
    df = df.loc[:,colsToKeep]
    df["Total Diameter"] = 0
    df["Species"] = speciesName
    df.set_index(["PlotID","Species","ObservationID"],inplace=True)
    df = df.sort_index()
    allPIDs = df.groupby("PlotID")
    for pid, gr in allPIDs:
        #gr will contain two observations
        hi = li = 0
        if gr.iloc[0].NumIndividuals > gr.iloc[1].NumIndividuals:
            hi, li = 0,1
        elif gr.iloc[0].NumIndividuals < gr.iloc[1].NumIndividuals:
            hi, li = 1,0

        if hi != li:
            #Increase in individuals
            df.loc[gr.index[hi],"NumIndividuals"] = randint(10,20)
            df.loc[gr.index[li],"NumIndividuals"] = randint(1,9)
            k = 0

    #we need to replace 0-1 pairs with random numbers of individuals so that they are tracked in each of the plots
    k = 0


    if append and os.path.exists(PATHS.Occ.Combined):
        dfOld = pd.read_csv(PATHS.Occ.Combined)
        dfOld.set_index(["PlotID","Species","ObservationID"],inplace=True)
        df = pd.concat([df,dfOld])
        #ensure no NAs
        assert np.all(df.notna()), "Some n/a values created"

    pyutil.writePandasToCSV(df,PATHS.Occ.Combined, index=True)

def _drawFromDist(probDensity, values, setToDrawFrom, numDraws):

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(p_normalized := probDensity / np.sum(probDensity))

    # Interpolating the sampling probabilities for each value in V
    sampling_probabilities = np.interp(setToDrawFrom, values, p_normalized)

    # Normalize the sampling probabilities for V
    sampling_probabilities /= np.sum(sampling_probabilities)

    # Draw 50 samples from V based on the computed probabilities
    sampleIndices = np.random.choice(range(0,len(setToDrawFrom)), size=numDraws, p=sampling_probabilities, replace=False)

    return setToDrawFrom.index[sampleIndices]


def generateVirtualSpecies(_gradient:str = "Elevation", shiftInStd:float = 0.1, widthInStd:float = 0.3, numOcc = 700, speciesName:str = None, plotResults:bool = True, appendResults:bool = True, plotParams:dict = None):
    """
    Will generate occurrences for virtual species moving up a given gradient. Movement happens around the mean by providing a distance in standard deviations across all plots.
    :param _gradient: The variable along which species will move
    :param shiftInStd: Species will start at mean - shiftInStd*std and move to mean + shiftInStd*std
    :param widthInStd: The width of the presence probability distribution in standard deviations of the total plot distribution.
    :param numOcc: Number of occurrences to generate, needs to be a few times smaller than the number of plots to have randomness in drawing this samples from available plots. if numOcc == numPlots the before and after distirbutions will be necessarily identical.
    :param speciesName: Name of species to use in the CSV file.
    :return:
    """
    
    termutil.chapPrint("Generating a synthetic species moving along the gradient %s by %.2f standard deviations" % (_gradient,shiftInStd))
    piS = pd.read_csv(PATHS.Virtual.VirtualPlotInfoWithProps)

    # get the distirbution along the gradient and plot it
    allVals = piS[_gradient].to_numpy()

    grRange = np.linspace(piS[_gradient].min(), piS[_gradient].max(), 100)
    kde = stats.gaussian_kde(piS[_gradient].to_numpy())
    gradientDist = kde(grRange)
    gradientDist /= gradientDist.max()

    std = piS[_gradient].std()
    mean = piS[_gradient].mean()

    #presence probability is a gaussian with a width of 0.3 std and a mean at the mean of the gradient
    presenceProbBefore = stats.norm.pdf(grRange, mean - std*shiftInStd/2, widthInStd*std)
    presenceProbAfter = stats.norm.pdf(grRange, mean + std*shiftInStd/2, widthInStd*std)

    presenceProbBefore /= presenceProbBefore.max()
    presenceProbAfter /= presenceProbAfter.max()


    piS["NumIndividuals"] = 0
    #generate a rnd num
    beforePlots = piS[piS.ObservationID > 0][_gradient]
    beforePresent  = _drawFromDist(presenceProbBefore, grRange, beforePlots, numOcc)
    piS.loc[beforePresent,"NumIndividuals"] = 1

    afterPlots = piS[piS.ObservationID < 0][_gradient]
    afterPresent  = _drawFromDist(presenceProbAfter, grRange, afterPlots, numOcc)
    piS.loc[afterPresent,"NumIndividuals"] = 1

    #Do a KDE of the actual presence probability
    kdeBef = stats.gaussian_kde(piS[(piS.ObservationID > 0) & (piS.NumIndividuals > 0) ][_gradient].to_numpy())
    kdeAfter = stats.gaussian_kde(piS[(piS.ObservationID < 0) & (piS.NumIndividuals > 0) ][_gradient].to_numpy())

    kdeBef = kdeBef(grRange)
    kdeAfter = kdeAfter(grRange)
    
    # compute the means of the kdes
    meanBef = np.sum(kdeBef * grRange) / np.sum(kdeBef)
    meanAfter = np.sum(kdeAfter * grRange) / np.sum(kdeAfter)
    
    print(f"    Species shifted along {_gradient} from: {meanBef:.2f} to: {meanAfter:.2f}. Difference: {meanAfter - meanBef:.2f}")


    kdeAfter /= kdeAfter.max()
    kdeBef /= kdeBef.max()
    #assign occurrence to each plot based on the presence probability

    if speciesName is None:
        speciesName = _gradient + " Species"
    _writeSpeciesOccurenceToCSV(piS, speciesName,appendResults)

    if plotResults:
        plotParams = plotParams if plotParams is not None else {}
        f = go.Figure()
        f.add_trace(go.Scatter(x=grRange, y=gradientDist, mode="lines", name=_gradient + " Distribution (all plots)", line=dict(width=1, color="black")))
        f.add_trace(go.Scatter(x=grRange, y=presenceProbBefore, mode="lines", name="Presence probability 2000", line=dict(dash="dot", width=1, color="red") ))
        f.add_trace(go.Scatter(x=grRange, y=presenceProbAfter, mode="lines", name="Presence probability 2019", line=dict(dash="dot", width=1, color="blue") ))
        f.add_trace(go.Scatter(x=grRange, y=kdeBef, mode="lines", name="Distribution of species 2000", line=dict(color="red")))
        f.add_trace(go.Scatter(x=grRange, y=kdeAfter, mode="lines", name="Distribution of species 2019", line=dict(color="blue")))
        f.update_layout(title=_gradient + " distribution of a synthetic species", xaxis_title=_gradient, yaxis_title="Density")
        #change xrange
        f.update_xaxes(**plotParams.get("xaxis",{}))
        f.update_yaxes(**plotParams.get("yaxis",{}))
        f.update_layout(**plotParams.get("layout",{}))
        # imgPath = lambda s: PATHS.plotFolderFigures + "Fig_SynthDist_%s.svg" % s
        # f = saveAsPrint(f"Fig_SynthDist_{speciesName}.svg",f,w="26%", h="40%", noLegend=True)
        f = saveAsPrint(f"Fig_SynthDist_{speciesName}.svg",f,w=450, h=440, noLegend=True)
        f.show()
    k = 0

def addClimateDataToVirtualPlotInfo(_overwrite:bool = False):
    """
    Will extract the climate data for each plot and save it, since this will be necessary to generate the virtual plot info.
    This step is part of the data generation process in main.py as well. If the species are generated here, these steps in main.py can be skipped.
    Set the _overwritePlotProps to False if you want to skip this step in main.py
    """
    
    termutil.chapPrint("Adding (real) climate data to virtual plots.")
    
    extractGeoLayers(_overwrite)
    extractClimateLayers(_overwrite)
    extractClimateLayersLinAppx(_overwrite)

    plotsWithGeoProps = pd.read_csv(PATHS.PlotInfo.WithGeoProps)
    climatePlots = pd.read_csv(PATHS.Bioclim.AtPlotsByYearLinearAppx)

    #get all predictor variable folders
    predFolder = PATHS.Raw.Predictors
    #get all Folders
    climVars = [f for f in os.listdir(predFolder) if os.path.isdir(predFolder+f)]
    allTuples = plotsWithGeoProps.groupby(["PlotID", "Year"])

    for (pid, y), gr in tqdm(allTuples):
        r = climatePlots[(climatePlots.PlotID == pid) & (climatePlots.Year == y) ]
        r = r.loc[:, climVars]
        #sort the columns of r so they are in the same order as columns
        if len(r) == 0:
            print(f"Warning: No climate data found for PlotID {pid} in year {y}. Check if the climate data is available for this year.")
            continue
        climVals = r.iloc[0, :].to_list()

        # add to the shifts as columns to the dataframe
        plotsWithGeoProps.loc[gr.index, climVars] = climVals

    removedPlots = np.unique(np.where(plotsWithGeoProps < -1e20)[0])
    removedPlots2 = np.unique(np.where(plotsWithGeoProps > 1e20)[0])
    removedPlots = np.unique(np.concatenate([removedPlots, removedPlots2]))
    if len(removedPlots) > 0:
        plotsWithGeoProps = plotsWithGeoProps.drop(removedPlots).reset_index(drop=True)
        print(f"Removed {len(removedPlots)} plots with improbable values (> 1e20 or < -1e20). These are likely nans where some variable was not available.")
    #assert no na
    assert not plotsWithGeoProps.isna().any().any()

    print("Plots are generated. No NA values are present. However in GeoTIFFs nA values can be something else (e.g. -3.4028235e+38) Double check manually that this is not the case and all values are valid.")
    pyutil.writePandasToCSV(plotsWithGeoProps, PATHS.Virtual.VirtualPlotInfoWithProps, "Generating Virtual Plot Info With Climate Props file")



def generateRandomPlots(numPlots: int, incFactor: int = 2, showPlots: bool = False, overwrite: bool = False):
    """
    Will generate a number of random plot locations based on a KDE distribution.
    Then will create two sets of observations for 2000 and 2019 with the same plot locations.
    Will load the pH geotiff file to check which plots are on the map and get the max dimensions. pH is the property with the most "holes" since it is not defined for lakes, while other geo and clim properties are.
    :param numPlots: The number of plots to generate (number of observations will be double this number). Choose more plots than you need, as some might not have values in them if they fall unto pixels where some climatic or geo property is not defined (happens often for lakes/rivers and edaphic properties).
    :param incFactor: Since not all plots will be on the map, this will increase the number of samples from the KDE, that will then be filtered for overlapping plot locations as well as plots outside the map. If not enough plots can be generated this number will be doubled recursively. Might be useful if your KDE contains a lot of small islands with many points randomly falling just outside the boundary.
    :param showPlots: If true will display a plot of the KDE, and the generated plots on the map
    :return: None, output written directly to PlotInfo in the _input directory.
    """
    
    termutil.chapPrint("Generating random plots for synthetic species")
    
    if os.path.exists(PATHS.PlotInfo.NoProps) and not overwrite:
        termutil.successPrint("     Skipped generating random plots. File exists.")
        return
    
    
    # Read a geotiff file for dimensions and as a mask
    img = PATHS.Raw.Predictors + "pH.tif"
    img = rasterio.open(img)
    imgn: np.ndarray = img.read(1)

    maxX = img.shape[1]
    maxY = img.shape[0]


    # Load KDE for NZ plot distribution
    with open(PATHS.Virtual.PlotDistribution, "rb") as f:
        kde = pickle.load(f)
    x = np.linspace(0, maxX, int(maxX/3))
    y = np.linspace(0, maxY, int(maxY/3))
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)


    # sample numPlots from distribution
    sample = kde.resample(size=numPlots * incFactor)
    piS = pd.DataFrame(sample.T, columns=["mapX", "mapY"])

    nanval = imgn[0, 0]
    # get a binary mask
    mask = imgn != nanval

    # round the plots coordinates to pixels
    piS["mapX"] = np.round(piS.mapX).astype(int)
    piS["mapY"] = np.round(piS.mapY).astype(int)

    # remove plots outside of bounds
    piS = piS.loc[(piS.mapX >= 0) & (piS.mapX < mask.shape[1]) & (piS.mapY >= 0) & (piS.mapY < mask.shape[0])]

    # remove duplicates
    piS.drop_duplicates(inplace=True)

    # filter out plots not on the map
    piS = piS.loc[mask[piS.mapY, piS.mapX]]

    if len(piS) < numPlots:
        print("     Not enough plots on the map, trying again.")
        return generateRandomPlots(numPlots, incFactor * 2)

    # reduce to the desired number of plots
    piS = piS.iloc[:numPlots]


    if showPlots:
        pg = PlotlyGraph()
        #normalize Z to 0-1
        Z = 255 * (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z))
        pg[0] *= px.imshow(Z, x=x, y=y)
        pg[1] *= px.imshow(mask)
        pg[1] *= go.Scatter(x=piS.mapX, y=piS.mapY, mode="markers", marker=dict(size=5, color="red"))
        pg.show(cols=2)

    #Convert to needed format
    #PlotID,ObservationID,Year,mapX,mapY
    piS["PlotID"] = range(1, len(piS) + 1)
    piS["ObservationID"] = range(1, len(piS) + 1)
    piS["Year"] = GlobalParams.minYear
    #duplicate wth negative IDs for year 2019
    piS2 = piS.copy()
    piS2["Year"] = GlobalParams.maxYear - 1
    piS2["ObservationID"] = -piS2["ObservationID"]
    piS = pd.concat([piS, piS2])

    pyutil.writePandasToCSV(piS, PATHS.PlotInfo.NoProps, "Generated Syntehtic Plots")

if __name__ == "__main__":
    overwrite = False
    showPlots = False
    #Generate random plots - this wll overwrite the PlotInfo.csv file in the _data/_input folder
    # generateRandomPlots(2000, showPlots=showPlots,overwrite=overwrite)
    
    #Add climate data to the virtual plot info needed to generate the virtual species
    # addClimateDataToVirtualPlotInfo(overwrite)
    
    # #Generate the species occurrences, will overwrite the Occurrence.csv file in the _data/_input folder
    generateVirtualSpecies( "Elevation",shiftInStd=0.08,widthInStd=0.2, numOcc=1000, speciesName="Elevation Up Species", plotResults=showPlots, appendResults=False, plotParams={"xaxis": {"range": [0, 1350]}})
    generateVirtualSpecies("BIOEnd12",shiftInStd=0.08,widthInStd=0.2, numOcc=1000, speciesName="Precipitation Up Species", plotResults=showPlots, plotParams={"xaxis": {"range": [0, 4500]}})
    generateVirtualSpecies("Latitude", shiftInStd=0.25,widthInStd=0.2, numOcc=1000, speciesName="Latitude Up Species", plotResults=showPlots, plotParams={"xaxis": {"range": [-47, -37]}})
