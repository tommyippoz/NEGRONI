import copy

import numpy

from negroni.classifiers.NEGRONILearner import NEGRONILearner

from sklearn.utils.random import sample_without_replacement

from negroni.utils.negroni_utils import get_name


class BaggingLearner(NEGRONILearner):

    def __init__(self, estimator, n_ensembles=10, bagging_ratio=None):
        super().__init__()
        self.estimator = estimator
        if n_ensembles > 1:
            self.n_ensembles = n_ensembles
        else:
            print("Ensembles have to be at least 2")
            self.n_ensembles = 10
        if bagging_ratio is not None:
            self.bagging_ratio = bagging_ratio
        else:
            self.bagging_ratio = 1 / n_ensembles**(1/10)
        self.base_learners = []

    def classifier_fit(self, train_features, train_labels):

        bootstrap = []
        train_n = len(train_features)
        samples_n = int(train_n*self.bagging_ratio)
        for i in range(self.n_ensembles):
            indexes = sample_without_replacement(n_population=train_n, n_samples=samples_n)
            if train_labels is not None:
                bootstrap.append([numpy.asarray([train_features[i] for i in indexes]),
                                 numpy.asarray([train_labels[i] for i in indexes])])
            else:
                bootstrap.append([[train_features[i] for i in indexes], None])

        for [sample_x, sample_y] in bootstrap:
            learner = copy.deepcopy(self.estimator)
            if sample_y is not None:
                learner.fit(sample_x, sample_y)
            else:
                learner.fit(sample_x)
            self.base_learners.append(learner)

    def classifier_predict_proba(self, test_features):
        proba = numpy.zeros((len(test_features), len(self.classes_)))
        for clf in self.base_learners:
            predictions = clf.predict_proba(test_features)
            for i in range(len(test_features)):
                proba[i] += predictions[i]
        return proba / self.n_ensembles

    def classifier_predict(self, test_features):
        proba = self.predict_proba(test_features)
        return numpy.argmax(proba, axis=1)

    def get_name(self):
        return "Bagging(" + str(self.n_ensembles) + "-" + get_name(self.estimator) + ")"
