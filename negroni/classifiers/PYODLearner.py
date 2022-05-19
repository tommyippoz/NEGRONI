from negroni.classifiers.NEGRONIWrapper import NEGRONIWrapper


class PYODLearner(NEGRONIWrapper):

    def __init__(self, estimator):
        super().__init__(estimator)

    def classifier_predict_proba(self, test_features):
        proba = self.estimator.predict_proba(test_features)
        pred = self.estimator.predict(test_features)
        for i in range(len(pred)):
            min_p = min(proba[i])
            max_p = max(proba[i])
            proba[i][pred[i]] = max_p
            proba[i][1-pred[i]] = min_p
        return proba

