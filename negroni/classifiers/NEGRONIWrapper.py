from negroni.classifiers.NEGRONILearner import NEGRONILearner

from sklearn.base import BaseEstimator

from negroni.utils.negroni_utils import get_name


class NEGRONIWrapper(NEGRONILearner):

    def __init__(self, estimator, verbose=False):
        super().__init__(verbose)
        self.estimator = estimator

    def classifier_fit(self, train_features, train_labels):
        self.estimator.fit(train_features, train_labels)

    def classifier_predict(self, test_features):
        return self.estimator.predict(test_features)

    def classifier_predict_proba(self, test_features):
        return self.estimator.predict_proba(test_features)

    def get_name(self):
        return get_name(self.estimator)
