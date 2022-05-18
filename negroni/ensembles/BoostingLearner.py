import copy

import numpy

from negroni.classifiers.NEGRONILearner import NEGRONILearner

from sklearn.utils.random import sample_without_replacement

from negroni.utils.negroni_utils import get_name, sample_weight


class BoostingLearner(NEGRONILearner):

    def __init__(self, estimator, n_ensembles=10, learning_rate=None, sampling_ratio=None):
        super().__init__()
        self.estimator = estimator
        if n_ensembles > 1:
            self.n_ensembles = n_ensembles
        else:
            print("Ensembles have to be at least 2")
            self.n_ensembles = 10
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 2
        if sampling_ratio is not None:
            self.sampling_ratio = sampling_ratio
        else:
            self.sampling_ratio = 1 / n_ensembles**(1/2)
        self.base_learners = []

    def classifier_fit(self, train_features, train_labels):

        train_n = len(train_features)
        samples_n = int(train_n*self.sampling_ratio)
        weights = numpy.full(train_n, 1/train_n)
        for i in range(self.n_ensembles):
            indexes = sample_weight(weights, samples_n)
            sample_x = numpy.asarray([train_features[i] for i in indexes])
            sample_y = numpy.asarray([train_labels[i] for i in indexes])
            learner = copy.deepcopy(self.estimator)
            if sample_y is not None:
                learner.fit(sample_x, sample_y)
            else:
                learner.fit(sample_x)
            self.base_learners.append(learner)

            # Update Weights
            pred_y = learner.predict(train_features)
            update_flag = numpy.where(train_labels == pred_y, 0, 1)
            weights = weights*(1+self.learning_rate*update_flag)
            weights = weights/sum(weights)

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
        return "Boosting(" + str(self.n_ensembles) + "-" + get_name(self.estimator) + ")"
