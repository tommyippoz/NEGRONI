import numpy

from negroni.classifiers.NEGRONILearner import NEGRONILearner


def get_name(learner):
    if isinstance(learner, NEGRONILearner):
        return learner.get_name()
    else:
        return learner.__class__.__name__


def sample_weight(weights, samples_n):
    indexes = numpy.random.choice(len(weights), samples_n, replace=False, p=weights)
    return indexes
