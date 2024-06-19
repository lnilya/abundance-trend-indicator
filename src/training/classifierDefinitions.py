import copy
import math

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading

from GlobalParams import GlobalParams
from src.classes.Enums import ModelType
from src.classes.VariableList import VariableList


## Main settings for the classifier hyper parameter grid search


kernel_options = ([RBF(length_scale=l, length_scale_bounds=(1e-5, 1e7)) for l in [0.01, 0.1, 1, 10, 50]] +
                  [Matern(length_scale=l, nu=n, length_scale_bounds=(1e-5, 1e7)) for l in [0.1, 1, 10] for n in
                   [0.5, 1.5, 2.5]] +
                  [RationalQuadratic(length_scale=l, alpha=a, alpha_bounds=(1e-5, 1e7)) for l in [0.1, 1, 10, 100] for a
                   in [0.01, 0.1, 1, 10, 100]])

_score = "roc_auc"

classifiers = {
    ModelType.ANN1: #ANN with single hidden layer
        GridSearchCV(MLPClassifier(solver="adam", max_iter=50000),
                     param_grid={
                         'hidden_layer_sizes': [(1/36),(0.05), (0.075), (0.1), (0.125), (0.15), (0.25), (0.5)],
                         # in % of the input layers, will be transformed before use.
                         'activation': ['tanh', 'relu'],
                         'alpha': [0.0001, 0.001, 0.01, 0.1],
                         'learning_rate': ['constant', 'invscaling', 'adaptive'],
                         'learning_rate_init': [0.001, 0.01, 0.1]
                     },
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.ANN2: #ANN with 2 layers
        GridSearchCV(MLPClassifier(solver="adam", max_iter=50000),
                     param_grid={
                         'hidden_layer_sizes': [(0.05, 0.05),
                                                (0.1, 0.05), (0.1, 0.1),
                                                (0.15, 0.05), (0.15, 0.1), (0.15, 0.15),
                                                (0.25, 0.05), (0.25, 0.1), (0.25, 0.15),(0.25, 0.25),
                                                (0.5, 0.05), (0.5, 0.1), (0.5, 0.15),(0.5, 0.25),(0.5, 0.5),
                                                (0.75, 0.05), (0.75, 0.1), (0.75, 0.15),(0.75, 0.25),(0.75, 0.5),(0.75, 0.75)
                                                ],
                         # in % of the input layers, will be transformed before use.
                         'activation': ['tanh', 'relu'],
                         'alpha': [0.0001, 0.001, 0.01, 0.1],
                         'learning_rate': ['constant', 'invscaling', 'adaptive'],
                         'learning_rate_init': [0.001, 0.01, 0.1]
                     },
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.SVMW:
        GridSearchCV(svm.SVC(kernel='rbf', class_weight="balanced"),
                     param_grid={"C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000],
                                 "gamma": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000, "scale", "auto"]},
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.SVM:
        GridSearchCV(svm.SVC(kernel='rbf'),
                     param_grid={"C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000],
                                 "gamma": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000, "scale", "auto"]},
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.SVMPoly:  # incredibly slow...
        GridSearchCV(svm.SVC(kernel='poly'),
                     param_grid={"C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000],
                                 "gamma": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000, "scale", "auto"]},
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.SVMSig:
        GridSearchCV(svm.SVC(kernel='sigmoid'),
                     param_grid={"C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000],
                                 "gamma": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000, "scale", "auto"]},
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.KNN:
        GridSearchCV(KNeighborsClassifier(),
                     param_grid={"n_neighbors": [2, 3, 4, 5, 7, 10],
                                 "weights": ["uniform", "distance"]},
                     scoring=_score,
                     cv=5, n_jobs=11),
    ModelType.GP: GridSearchCV(GaussianProcessClassifier(max_iter_predict=50000),
                               {'kernel': kernel_options},
                               cv=5, scoring=_score,
                               n_jobs=6),

    ModelType.GLM: GridSearchCV(LogisticRegression(class_weight="balanced", max_iter=50000),
                                param_grid={"C": [1, 100, 1000, 10000, 50000, 100000, 500000, 1000000]},
                                scoring=_score,
                                cv=5, n_jobs=8),

    ModelType.RF: GridSearchCV(RandomForestClassifier(),
                               param_grid={"max_depth": [4, 5, 6, 7, 8, 9, 10],
                                           "n_estimators": [50, 100, 200, 300, 400, 500],
                                           'min_samples_split': [2, 5, 10]},
                               scoring=_score,
                               cv=5, n_jobs=11),

    ModelType.LSKNN: GridSearchCV(LabelSpreading(kernel='knn', max_iter=1000),
                                  param_grid={"alpha": [0.1, 0.25, 0.5, 0.75, 0.9], "n_neighbors": [2, 3, 4, 5, 7, 10]},
                                  scoring=_score,
                                  cv=5, n_jobs=11),
    ModelType.LSRBF: GridSearchCV(LabelSpreading(kernel='rbf', max_iter=1000),
                                  param_grid={"alpha": [0.1, 0.25, 0.5, 0.75, 0.9]},
                                  scoring=_score,
                                  cv=5, n_jobs=11)
}


def getClassifier(t: ModelType, vars: VariableList):
    sv = classifiers[t]
    maxJobs = math.prod([len(v) for v in sv.param_grid.values()]) * sv.cv
    sv.n_jobs = min(maxJobs, GlobalParams.parallelProcesses)

    if t == ModelType.ANN1 or t == ModelType.ANN2:
        clf = copy.deepcopy(sv)
        hiddenLayerRel = clf.param_grid["hidden_layer_sizes"]
        newSizes = []
        for i, h in enumerate(hiddenLayerRel):
            if isinstance(h, float):
                newSizes.append(max(1, math.ceil(h * len(vars))))
            else:
                newSizes.append(tuple([max(1, math.ceil(hin * len(vars))) for hin in h]))

        newSizes = list(set(newSizes))

        clf.param_grid["hidden_layer_sizes"] = newSizes
        return clf

    return sv
