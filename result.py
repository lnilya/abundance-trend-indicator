from GlobalParams import GlobalParams
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


    corrs = dict(zip(_vars.list, corrs))

    return corrs


def run(_species):
    _comb = Dataset.AdultsOnly
    _vars = PredictiveVariableSet.Full
    _cp = ClassificationProblem.IncDec
    _model = ModelType.GLM
    _nr = 0
    _nrb = NoiseRemovalBalance.Equal
    _years = list(range(GlobalParams.minYear, GlobalParams.maxYear-1))

    mid = ModelFileID(_species, _model, _vars, _cp, _comb, 0, _nrb)
    model = TrainedModel.load(mid.file)
    print(f"Loaded Model {_model.value} for {_species}. Test score: {model.testScore}")


    corrs = getCorrelation(_species, _comb,_vars,_cp,_model,_nr,_nrb,_years)

    #rename the keys
    corrs = {names.get(k,k):v for k,v in corrs.items()}

    #plot as bar
    px.bar(x=corrs.keys(), y=corrs.values(), title=f"Correlation between ATI and Environmental Variables for {_species}").show()

    mmpf = ModelMeanPredictionFileID(_years, mid)
    simf = SimilarityDataFileID.initFromPrediction(mmpf,  GlobalParams.similarity_metric, GlobalParams.similarity_k)
    sim = SimilarityData.load(simf.file)
    mmp = ModelMeanPrediction.load(mmpf.file)
    # sim.plotData()
    mmp.plotData()

    #compute correlation between the ATI values and environmental variables


    pass


if __name__ == "__main__":
    run("Precipitation Up Species")
    run("Elevation Up Species")
