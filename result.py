import pandas as pd

from GlobalParams import GlobalParams
from src.__libs.plotutil import saveAsPrint
from src.classes.Enums import ModelType, PredictiveVariableSet, ClassificationProblem, Dataset, NoiseRemovalBalance
from src.classes.FileIDClasses import ModelMeanPredictionFileID, ModelFileID, SimilarityDataFileID, StackedDataFileID
from src.classes.FlatMapData import FlatMapData
from src.classes.ModelMeanPrediction import ModelMeanPrediction
from src.classes.SimilarityData import SimilarityData
from src.classes.StackedData import StackedData
from src.classes.TrainedModel import TrainedModel
import numpy as np
import plotly.express as px

from src.classes.VariableList import VariableList

names = {
    "BIOEnd1":  "Annual Mean Temperature",
    "BIOEnd2":  "Mean Diurnal Range ",
    "BIOEnd3":  "Isothermality",
    "BIOEnd4":  "Temperature Seasonality",
    "BIOEnd5":  "Max Temperature of Warmest Month",
    "BIOEnd6":  "Min Temperature of Coldest Month",
    "BIOEnd7":  "Temperature Annual Range",
    "BIOEnd8":  "Mean Temperature of Wettest Quarter",
    "BIOEnd9":  "Mean Temperature of Driest Quarter",
    "BIOEnd10": "Mean Temperature of Warmest Quarter",
    "BIOEnd11": "Mean Temperature of Coldest Quarter",
    "BIOEnd12": "Annual Precipitation",
    "BIOEnd13": "Precipitation of Wettest Month",
    "BIOEnd14": "Precipitation of Driest Month",
    "BIOEnd15": "Precipitation Seasonality",
    "BIOEnd16": "Precipitation of Wettest Quarter",
    "BIOEnd17": "Precipitation of Driest Quarter",
    "BIOEnd18": "Precipitation of Warmest Quarter",
    "BIOEnd19": "Precipitation of Coldest Quarter"
}
def getCorrelation(_species:str, _dataset:Dataset, _vars:VariableList, _cp:ClassificationProblem, _model:ModelType, _nr:float, _nrb:NoiseRemovalBalance, _years2:list):
    """
    Estimates the correlation between ATI values and environmental variables. The correlation is computed by correlating the mean ATI values with the mean environmental variable values over the time period.
    Correlaton is only estimated where predictions for the species are available.
    :param _species: Species name
    :param _dataset: Class combination
    :param _vars: Predictive variable set
    :param _cp: Classification problem (Inc or Dec)
    :param _model: Which model's predictions will be used
    :param _nr: Noise reduction value used for the predictions
    :param _nrb: Noise removal balance type used for the predictions
    :param _years2: Years to consider
    :return:
    """
    _years = _years2.copy()
    _years2[-1] += 1

    _simRangeCutoff = 1.2

    #Stack the predictive variuables
    data = []
    nanmask = None

    for y in _years2:
        sdid = StackedDataFileID(y, PredictiveVariableSet.Full)
        sd = StackedData.load(sdid.file)
        if nanmask is None:
            nanmask = sd._nanMask
        data.append(sd.getDataAs1DArray(subset=_vars))

    data = np.array(data).mean(axis=0)
    stackedData = FlatMapData(None, data, _vars.list, nanmask)



    #Load all predictions and compute correlations
    mid = ModelFileID(_species, _model, _vars, _cp, _dataset, _nr, _nrb)
    res = []
    classifier = TrainedModel.load(mid.file)
    if classifier.testScore < GlobalParams.predictionThreshold:
        return None

    #load similarity and efti
    simid = SimilarityDataFileID(_years2, GlobalParams.similarity_k, GlobalParams.similarity_metric, _species, PredictiveVariableSet.Full,
                                 _cp, _dataset)
    sim = SimilarityData.load(simid.file)
    mmp = ModelMeanPrediction.load(ModelMeanPredictionFileID(_years2, classifier._fileID).file)

    mask1D = sim.getMask1D(_simRangeCutoff)
    efti1D = mmp.getDataAs1DArray(mask1D=mask1D)[:, 0]
    predData = stackedData.getDataAs1DArray(mask1D=mask1D)

    corrs = []
    for i in range(predData.shape[1]):
        corrs.append(np.corrcoef(efti1D, predData[:, i])[0, 1])

    # rename the keys
    varNames = [names.get(v,v) for v in _vars.list]
    corrs = pd.DataFrame({'Species':_species,
                          'Model':_model.value,
                          'Variable': varNames,
                          'Correlation': corrs,
                          'ClassCombinationMethod':_dataset.value,
                          'NoiseReduction':_nr, 'NoiseReductionBalance':_nrb.value, 'Problem':_cp.value, 'SimRangeCutoff':_simRangeCutoff})
    # corrs = dict(zip(_vars.list, corrs))

    return corrs

def showEnvironmentalConditions(vars:list[str], _years):
    data = np.ndarray((1476,1003,len(vars),len(_years)))
    nanmask = None

    for e,y in enumerate(_years):
        sdid = StackedDataFileID(y, PredictiveVariableSet.Full)
        sd = StackedData.load(sdid.file)

        data[:,:,:,e] = sd.getDataAs2DArray(vars)

    datamean = np.mean(data,axis=3)

    for i,v in enumerate(vars):
        f = px.imshow(datamean[:,:,i], color_continuous_scale=px.colors.sequential.Turbo, title=f"{v}")
        # f = px.imshow(fullImg,color_continuous_scale=px.colors.sequential.RdBu_r, title=f"ATI prediction for {s}")
        f.update_xaxes(visible=False)
        f.update_yaxes(visible=False)
        f = saveAsPrint(f"Fig_SynthRes_{v}.svg",f, w="70%")
        f.show()
    k = 0

def run(_species, exportSVGFiles:bool = False):
    _comb = Dataset.AdultsOnly
    _vars = PredictiveVariableSet.Full
    _cp = ClassificationProblem.IncDec
    _model = ModelType.GLM
    _nr = 0
    _nrb = NoiseRemovalBalance.Equal
    _years = list(range(GlobalParams.minYear, GlobalParams.maxYear-1))


    allCorrs = None
    for s in _species:
        mid = ModelFileID(s, _model, _vars, _cp, _comb, 0, _nrb)
        model = TrainedModel.load(mid.file)
        print(f"Loaded Model {_model.value} for {s}. Test score: {model.testScore}")
        corrs = getCorrelation(s, _comb,_vars,_cp,_model,_nr,_nrb,_years.copy())
        if allCorrs is None:
            allCorrs = corrs
        else:
            allCorrs = pd.concat([allCorrs, corrs], ignore_index=True)

        mmpf = ModelMeanPredictionFileID([GlobalParams.minYear, GlobalParams.maxYear-1], mid)
        simf = SimilarityDataFileID.initFromPrediction(mmpf, GlobalParams.similarity_metric, GlobalParams.similarity_k)
        sim = SimilarityData.load(simf.file)
        mmp = ModelMeanPrediction.load(mmpf.file)
        if not exportSVGFiles:
            sim.plotData()
            mmp.plotData()
        else:
            fullImg = mmp.plotData(returnOnly=True)
            f = px.imshow(fullImg,color_continuous_scale=px.colors.sequential.RdBu_r, title=f"ATI prediction for {s}")
            f.update_xaxes(visible=False)
            f.update_yaxes(visible=False)
            f = saveAsPrint(f"Fig_SynthRes_{s}.svg",f, w="70%")
            f.show()

    #Show the ATI correlations
    f = px.bar(allCorrs, x="Variable", y="Correlation", color="Species", title=f"Correlation between ATI and Environmental Variables", barmode="group")
    f.update_yaxes(title_text=f"Correlation {_species}")
    if exportSVGFiles:
        f = saveAsPrint("Fig_SynthRes_Corr.svg",f, w="118%", noLegend=True)
    f.show()

if __name__ == "__main__":
    #BIO12 is the variable for total annual precipitation
    # showEnvironmentalConditions(["Elevation","BIOEnd12"],range(GlobalParams.minYear, GlobalParams.maxYear-1))
    run(["Elevation Up Species","Precipitation Up Species"],True)
