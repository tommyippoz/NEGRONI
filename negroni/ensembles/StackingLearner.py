import copy

import numpy
import pandas

from negroni.classifiers.NEGRONILearner import NEGRONILearner
from negroni.utils.negroni_utils import get_name


class StackingLearner(NEGRONILearner):

    def __init__(self, base_level_learners, meta_level_learner, use_training=False):
        super().__init__()
        self.base_level_learners = base_level_learners
        self.meta_level_learner = meta_level_learner
        self.use_training = use_training
        self.stacking_data = None

    def classifier_fit(self, train_features, train_labels):

        stacking_data = []
        stacking_columns = []

        # Training base-level-learners
        for li in range(len(self.base_level_learners)):
            learner_name = get_name(self.base_level_learners[li])
            try:
                self.base_level_learners[li].fit(train_features, train_labels)
                learner_proba = self.base_level_learners[li].predict_proba(train_features)
                stacking_data.append(learner_proba)
                stacking_data.append([[0 if learner_proba[i][0] > learner_proba[i][1] else 1]
                                      for i in range(len(learner_proba))])
                stacking_columns.extend([learner_name + "_normal",
                                         learner_name + "_anomaly",
                                         learner_name + "_label"])
            except:
                print("Execution of learner " + learner_name + " failed")
                self.base_level_learners[li] = None
        self.base_level_learners = [i for i in self.base_level_learners if i]

        stacking_data = numpy.concatenate(stacking_data, axis=1)
        self.stacking_data = pandas.DataFrame(data=copy.deepcopy(stacking_data),
                                              columns=stacking_columns)
        if self.use_training:
            stacking_data = numpy.concatenate([train_features, stacking_data], axis=1)

        # Trains meta-level learner
        self.meta_level_learner.fit(stacking_data, train_labels)
        self.stacking_data["bin_label"] = train_labels

    def classifier_predict_proba(self, test_features):
        stacking_data = []

        # Scoring base-level-learners
        for learner in self.base_level_learners:
            learner_proba = learner.predict_proba(test_features)
            stacking_data.append(learner_proba)
            stacking_data.append([[0 if learner_proba[i][0] > learner_proba[i][1] else 1]
                                  for i in range(len(learner_proba))])

        stacking_data = numpy.concatenate(stacking_data, axis=1)
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
