from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score


class PretrainedVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, voting='hard'):
        self.classifiers = classifiers
        self.voting = voting

    def __getitem__(self, item):
        return self.classifiers[item]

    def predict(self, X):
        """ Predict class labels for X as a majority vote of the classifiers."""
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        maj_vote = np.sum(predictions, axis=0) >= (len(self.classifiers) / 2)

        if len(self.classifiers) % 2 == 0:
            raise ValueError("Voting is ambiguous with an even number of classifiers.")

        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X as a mean between the 1/0 outputs of the classifiers."""
        probas = np.array([clf.predict(X) for clf in self.classifiers])
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def fit(self, X, y):
        """ This method is not implemented, as the classifiers are already fitted. """
        raise NotImplementedError("This ensemble uses pre-trained classifiers and does not support fitting.")

    def score(self, X, y, scoring=None):
        """ Returns the score using the scoring function provided. Default is accuracy. """
        predictions = self.predict(X)
        if scoring:
            return scoring(y, predictions)
        return accuracy_score(y, predictions)

    @property
    def feature_importances_(self):
        #For RF classifiers only the mean of the feature importances is returned
        if hasattr(self.classifiers[0], "feature_importances_"):
            return np.mean([s.feature_importances_ for s in self.classifiers], axis=0)
