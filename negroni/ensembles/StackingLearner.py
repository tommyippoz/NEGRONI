import numpy
from sklearn.utils.validation import check_is_fitted

from negroni.classifiers.NEGRONILearner import NEGRONILearner
from negroni.utils.negroni_utils import get_name


class StackingLearner(NEGRONILearner):

    def __init__(self, base_level_learners, meta_level_learner, use_training=False):
        super().__init__()
        self.base_level_learners = base_level_learners
        self.meta_level_learner = meta_level_learner
        self.use_training = use_training

    def classifier_fit(self, train_features, train_labels):

        stacking_data = []

        # Training base-level-learners
        for learner in self.base_level_learners:
            learner.fit(train_features, train_labels)
            stacking_data.append(learner.predict_proba(train_features))

        stacking_data = numpy.concatenate(stacking_data, axis=1)
        if self.use_training:
            stacking_data = numpy.concatenate([train_features, stacking_data], axis=1)

        # Trains meta-level learner
        self.meta_level_learner.fit(stacking_data, train_labels)

    def classifier_predict_proba(self, test_features):
        stacking_data = []

        # Training base-level-learners
        for learner in self.base_level_learners:
            stacking_data.append(learner.predict_proba(test_features))

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
