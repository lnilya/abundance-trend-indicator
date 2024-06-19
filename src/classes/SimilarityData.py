from typing import Union

import numpy as np

from src.classes.Enums import PredictiveVariableSet, ClassificationProblem, ClassCombinationMethod
from src.classes.FileIDClasses import SimilarityDataFileID
from src.classes.FlatMapData import FlatMapData
import plotly.express as px

import scipy.stats as stats
from src.datautil import datautil

import pandas as pd
import paths as PATHS

class SimilarityData(FlatMapData):
    """
    For prediction every pixel on the NZ map is moved into an array and the 37 (or so features) measured.
    This yields a NxMxF array where where NxM is the size of the map and F the number of features.
    The array is then flattened to NMxF and NaNs (majority of the pixels) are removed.
    Where the nans occurred is stored in the nanmask, this allows it to transform the prediction back into the original shape
    and display as a map.

    Each StackedData is specific to a year as climate is year dependent.
    """

    _normalizationQuantile = None

    def __init__(self,id:SimilarityDataFileID, data:Union[np.ndarray, dict[str,np.ndarray]], nanMask:np.ndarray = None, shape = (1476,1003), _normQuantile = None):
        # stack the data into a numpy array if it comes as dict
        if isinstance(data, dict):
            # Final end result: For each year a list of 2D arrays sorted in the same way as variables, ready to be used for prediction
            dataAsList = [data[k] for k in data.keys()]
            data = np.stack(dataAsList, axis=1)

        if data.shape[1] != 1:
            data = np.mean(data, axis=1).reshape(-1,1)

        self._normalizationQuantile = _normQuantile
        super().__init__(id,data, ["Similarity"],nanMask,shape,np.float32)

    def computeNormalization(self, occData = None, quantile = 0.95):
        """The idea is to extract the distances needed to include 95% of training or occurrence data.
        The noramlization will set this value in the similarity map to 1 and scale the rest accordingly. This way we can always
        ensure that the similarity surface includes all occurrences."""

        if occData is None:
            occData = pd.read_csv(PATHS.Occ.Combined, usecols=["ParentPlotID", "Species"])
            occData, _ = datautil.getAllPlotInfo(False, ["mapX", "mapY"], occData)

        occData = occData.loc[occData.Species == self._fileID["Species"]]
        occData = occData.drop_duplicates()  # we are only interested in plots

        img = self.getDataAs2DArray(["Similarity"])[:,:,0]

        if self._normalizationQuantile is not None:
            print(f"Data is already normalized at ({self._normalizationQuantile}).")

        occVals = img[occData.mapY, occData.mapX]
        occVals = occVals[~np.isnan(occVals)]
        self._normalizationQuantile = np.quantile(occVals, quantile)

        return self._normalizationQuantile

    def displayDistHistograms(self, img = None, occ = None, trData = None, _q = 0.95, _subsample = 1, returnOnly = False, showTrainingSet:bool = True):

        if img is None:
            img = self.getImage(False,_subsample)
        if occ is None:
            trData, occ = self._loadOccData(_subsample)


        trVals = img[trData.mapY, trData.mapX]
        occVals = img[occ.mapY, occ.mapX]
        # remove nans
        trVals = trVals[~np.isnan(trVals)]
        occVals = occVals[~np.isnan(occVals)]

        # rune a KDE on the data
        kde0 = stats.gaussian_kde(trVals)
        kde1 = stats.gaussian_kde(occVals)
        x = np.linspace(0, np.nanmax(img), 400)

        # determine 95% quantile for the x-axis
        trQ95 = np.quantile(trVals, _q)
        occQ95 = np.quantile(occVals, _q)

        rangeX = [0,2*max(trQ95,occQ95)]

        df = pd.DataFrame({"Distance": x, "Training": kde0(x), "Occurrence": kde1(x)})
        # move Training and Occurrence to the same columns
        df = pd.melt(df, id_vars=["Distance"], value_vars=["Training", "Occurrence"], var_name="Type",
                     value_name="Frequency")

        if not showTrainingSet:
            df = df[df.Type == "Occurrence"]

        f2 = px.line(df, x="Distance", y="Frequency", color="Type", title="Distribution of Similarity Values", range_x=rangeX)






        if showTrainingSet:
            f2.add_shape(type="line", x0=trQ95, x1=trQ95, y0=0, y1=max(df.Frequency),
                         line=dict(color="blue", width=1, dash="dash"))
            f2.add_annotation(x=trQ95, y=max(df.Frequency), text=f"{trQ95:.2f}", showarrow=False)

        f2.add_shape(type="line", x0=occQ95, x1=occQ95, y0=0, y1=max(df.Frequency),
                     line=dict(color="red", width=1, dash="dash"))
        # add text annotations at top of line with x value
        f2.add_annotation(x=occQ95, y=max(df.Frequency), text=f"{occQ95:.2f}", showarrow=False)

        # change barmode
        f2.update_layout(title=f"Distribution of Similarity Values for {self._fileID['Species']} with {_q*100:.1f}% cutoff",)
        if not returnOnly:
            f2.show()

        return occQ95,trQ95,f2
    def _loadOccData(self, subsample:int):
        trData = pd.read_csv(PATHS.Shifts.allPlotsCombined(self._fileID["ClassCombinationMethod"]),
                           usecols=["Species", "Type", "ParentPlotID", "mapX", "mapY"])
        trData = trData[trData.Species == self._fileID["Species"]]

        occ = pd.read_csv(PATHS.Occ.Combined, usecols=["ParentPlotID", "Species"])
        occ = occ.loc[occ.Species == self._fileID["Species"]]
        occ = occ.drop_duplicates()  # we are only interested in plots
        occ, _ = datautil.getAllPlotInfo(False, ["mapX", "mapY"], occ)

        if subsample > 1:
            occ.mapX /= subsample
            occ.mapY /= subsample
            trData.mapX /= subsample
            trData.mapY /= subsample
            trData.mapX = trData.mapX.astype(int)
            trData.mapY = trData.mapY.astype(int)
            occ.mapX = occ.mapX.astype(int)
            occ.mapY = occ.mapY.astype(int)

        return trData, occ
    def plotData(self, subsample:int = 1):

        _rangeExpansion = 1.2

        img = self.displayImg({"Sim":self.getDataAs2DArray(["Similarity"])[:,:,0]}, False, subsample, True)



        #Show the plots
        trData, occ = self._loadOccData(subsample)

        #1. Display the dist histogram
        occQ95,trQ95,_ = self.displayDistHistograms(img, occ, trData)

        #Calculate the threshold image (2 colors for above and below the 100% of occurrence data)
        imgThreshold = img.copy()
        imgThreshold[imgThreshold > (occQ95 * _rangeExpansion)] = np.nanmax(img)
        imgThreshold[(imgThreshold >= occQ95)&(imgThreshold <= (occQ95 * _rangeExpansion))] = np.nanmax(img)/2
        imgThreshold[imgThreshold < occQ95] = 0

        #merge it with main image
        img = np.hstack((img, imgThreshold))

        #2. Display the maps

        f = px.imshow(img, color_continuous_scale=px.colors.sequential.thermal)

        f.add_scatter(x=occ["mapX"], y=occ["mapY"], mode="markers", marker=dict(size=4, color="lightgreen", symbol="cross"), hovertext=occ["ParentPlotID"])
        f.add_scatter(x=trData["mapX"], y=trData["mapY"], mode="markers", marker=dict(size=3, color="red"), hovertext=trData["ParentPlotID"])
        f.update_layout(title="Similarity Surface with %s"%self._fileID["Species"])


        f.show()

        #show the distribution of distances in training and occurrence data

    def getSurfaceAndTrainingData(self, showOcc:bool, showTrain:bool, subsample = 1, quantilleMax = 1):
        # Show the plots
        img = self.displayImg({"Sim": self.getDataAs2DArray(["Similarity"])[:, :, 0]}, False, subsample, True)
        trData, occ = self._loadOccData(subsample)
        img[img > np.nanquantile(img,quantilleMax)] = np.nanquantile(img,quantilleMax)
        f = px.imshow(img, color_continuous_scale=px.colors.sequential.YlGnBu_r)

        if showOcc:
            f.add_scatter(x=occ["mapX"], y=occ["mapY"], mode="markers",
                          marker=dict(size=4, color="orange", symbol="cross"), hovertext=occ["ParentPlotID"])
        if showTrain:
            f.add_scatter(x=trData["mapX"], y=trData["mapY"], mode="markers", marker=dict(size=3, color="red"),
                          hovertext=trData["ParentPlotID"])
        f.update_layout(title="Similarity Surface with %s" % self._fileID["Species"])
        return f

    def getMask1D(self, relCutoff = 1.2):
        mask = np.zeros_like(self._data, dtype=bool)
        mask[self._data < self._normalizationQuantile * relCutoff] = True
        return mask[:,0]
    def getMask(self, relCutoff = 1.2, subsample: int = 1):
        """Gets a 2D mask that only show the values themselves"""
        img = self.getImage(True, subsample)
        mask = np.zeros_like(img, dtype=bool)
        mask[img < relCutoff] = True
        return mask

    def getImage(self, normalized:bool = False, subsample: int = 1):
        img = super().getImage(False, None, subsample)
        if normalized:
            img /= self._normalizationQuantile

        return img

    def toDict(self) -> dict:
        d = super().toDict()
        d.update({"_normalizationQuantile":self._normalizationQuantile})
        return d

    @staticmethod
    def fromDict(dict: dict, type:any, fileID: SimilarityDataFileID) -> "SimilarityData":
        return SimilarityData(fileID,
                    dict["data"],
                    dict["nanMask"],
                    dict["shape"],
                    dict.get("_normalizationQuantile",None))
    @staticmethod
    def load(path) -> "SimilarityData":
        return FlatMapData.load(path, SimilarityData, SimilarityDataFileID)


if __name__ == "__main__":
    cp = ClassificationProblem.IncDec
    ccv = ClassCombinationMethod.AdultsOnly
    allVars = PredictiveVariableSet.Full
    sid = SimilarityDataFileID([1969,2019], 1,"l1","Myrsine australis", allVars, cp,
                               ccv)
    sd = SimilarityData.load(sid.file)
    sd.computeNormalization()
    sd.plotData()
    k = 0