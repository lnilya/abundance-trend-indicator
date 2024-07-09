import plotly.express as px

from src.classes.Enums import ModelType, PointDifficulty

class GlobalParams:

    """Global parameters for the entire project. Change anything here and it might have influence at various locations"""
    maxYear = 2020
    minYear = 1995
    yearRange = (minYear, maxYear)
    parallelProcesses = 8 #The number of parallel processes to use for the entire project. Should be ~the number of your CPUs
    minObservationsTotal = 100 #A species must have at least this many observations to be included in the analysis
    minObservationsPerClass = 20 #Each of the three classes (Inc,Dec,Same) should have at least that many observations

    maxYearDifference = 30 #The maximum number of years between observations. Will shrink dataset but too long observations do not have an influience anymore.

    testSetSize = 0.25 #Proportion of data used for testing
    testFolds = int(round(1/testSetSize)) #Proportion of data used for testing

    noise_k = 20 #Number of neighbours considered for noise analysis
    noise_alpha = 1.5 #Influence of distance for noise analysis
    noise_metric = "l1" #Metric for noise analysis, Generaly L1 is preferred for higher dimensions

    ## Similarity Surface
    similarity_k = 3 #Number of neighbours considered for similarity analysis
    similarity_metric = "l1" #Metric for similarity analysis, Generaly L1 is preferred for higher dimensions
    similairty_cutoff = 1.2 #Similairty of 1 means the area includes 95% of all trainingdata. We can go slightly above to have more area at the expense of the possibility that the classifiers do not predict as well.

    ## Predictions


    predictionThreshold = 0.7 #Only consider models with AUC scores >= this threshold


    floatFormat = "%.4f" #will make all the CSVs larger if adding more precision. But too little leads to duplicate datapoints sometimes.


    ## PLOTTING
    @staticmethod
    def modelColMapping():
        models = [ModelType.RF,ModelType.SVMW,ModelType.ANN1,ModelType.GLM]
        cols = {}
        for i, m in enumerate(models):
            cols[m] = px.colors.qualitative.Dark2[i]
            cols[m.value] = px.colors.qualitative.Dark2[i]
            me = m.parseToEnsembleVersion()
            cols[me] = px.colors.qualitative.Dark2[i]
            cols[me.value] = px.colors.qualitative.Dark2[i]

        return cols

    @staticmethod
    def pointColMapping():
        mp = {t.value: px.colors.qualitative.Set1[i] for i, t in enumerate(PointDifficulty)}
        mp[PointDifficulty.All.value] = px.colors.qualitative.Set1[-1] #gray
        return mp

