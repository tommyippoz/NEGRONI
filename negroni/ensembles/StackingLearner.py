import copy
import time

import numpy
import pandas

from negroni.classifiers.NEGRONILearner import NEGRONILearner
from negroni.utils.negroni_utils import get_name


class StackingLearner(NEGRONILearner):

    def __init__(self, base_level_learners, meta_level_learner,
                 use_training=False, train_meta=True, store_data=False, verbose=False):
        super().__init__(verbose)
        self.base_level_learners = base_level_learners
        self.meta_level_learner = meta_level_learner
        self.use_training = use_training
        self.store_data = store_data
        self.train_meta = train_meta
        self.stacking_features = None
        self.stacking_data = None
        self.stacking_test = None

    def classifier_fit(self, train_features, train_labels):

        stacking_data = []
        self.stacking_features = []

        # Training base-level-learners
        for li in range(len(self.base_level_learners)):
            learner_name = get_name(self.base_level_learners[li])
            try:
                start = time.time()
                self.base_level_learners[li].fit(train_features, train_labels)
                learner_proba = self.base_level_learners[li].predict_proba(train_features)
            except:
                print("Execution of learner " + learner_name + " failed")
                learner_proba = numpy.full((len(train_labels), 2), 0.5)

            stacking_data.append(learner_proba)
            stacking_data.append([[0 if learner_proba[i][0] >= learner_proba[i][1] else 1]
                                  for i in range(len(learner_proba))])
            self.stacking_features.extend([learner_name + "_normal",
                                     learner_name + "_anomaly",
                                     learner_name + "_label"])
            if self.verbose:
                print("Training of base-learner '" + learner_name + "' completed in " +
                      str(time.time() - start) + " sec")

        stacking_data = numpy.concatenate(stacking_data, axis=1)
        stacking_data = numpy.nan_to_num(stacking_data, nan=-0.5, posinf=0.5, neginf=0.5)

        if self.store_data:
            self.stacking_data = pandas.DataFrame(data=copy.deepcopy(stacking_data), columns=self.stacking_features)
            self.stacking_data["bin_label"] = train_labels

        if self.use_training:
            stacking_data = numpy.concatenate([train_features, stacking_data], axis=1)

        # Trains meta-level learner
        if self.train_meta:
            self.meta_level_learner.fit(stacking_data, train_labels)

    def classifier_predict_proba(self, test_features):
        stacking_data = []

        # Scoring base-level-learners
        for learner in self.base_level_learners:
            try:
                learner_proba = learner.predict_proba(test_features)
            except:
                learner_proba = numpy.full((len(test_features), 2), 0.5)
            stacking_data.append(learner_proba)
            stacking_data.append([[0 if learner_proba[i][0] > learner_proba[i][1] else 1]
                                  for i in range(len(learner_proba))])

        stacking_data = numpy.concatenate(stacking_data, axis=1)
        stacking_data = numpy.nan_to_num(stacking_data, nan=-0.5, posinf=0.5, neginf=0.5)

        if self.store_data:
            self.stacking_test = pandas.DataFrame(data=copy.deepcopy(stacking_data), columns=self.stacking_features)

        if self.use_training:
            stacking_data = numpy.concatenate([test_features, stacking_data], axis=1)

        return self.meta_level_learner.predict_proba(stacking_data)

    def classifier_predict(self, test_features):
        proba = self.predict_proba(test_features)
        return numpy.argmax(proba, axis=1)

    def get_name(self):
        return "Stacking(" + str(len(self.base_level_learners)) + "-" \
               + get_name(self.meta_level_learner) + ("-wTR)" if self.use_training else ")")

    def get_stacking_data(self):
        return self.stacking_data

    def get_stacking_test(self):
        return self.stacking_test
