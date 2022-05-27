import numpy
import pandas
from autogluon.tabular import TabularPredictor

from negroni.classifiers.NEGRONILearner import NEGRONILearner


class AutoGluonClassifier(NEGRONILearner):
    """
    Wrapper for classifiers taken from Gluon library
    clf_name options are
    GBM (LightGBM)
    CAT (CatBoost)
    XGB (XGBoost)
    RF (random forest)
    XT (extremely randomized trees)
    KNN (k-nearest neighbors)
    LR (linear regression)
    NN (neural network with MXNet backend)
    FASTAI (neural network with FastAI backend)
    """

    def __init__(self, label_name, clf_name, metric, verbose=False):
        super().__init__(verbose)
        self.model = TabularPredictor(label=label_name, eval_metric=metric)
        self.label_name = label_name
        self.feature_names = []
        self.clf_name = clf_name
        self.feature_importances_ = []

    def classifier_fit(self, x_train, y_train):
        if isinstance(x_train, pandas.DataFrame):
            self.feature_names = x_train.columns
        else:
            self.feature_names = ["feat_" + str(i) for i in range(0, x_train.shape[1])]
        df = pandas.DataFrame(data=x_train.copy(), columns=self.feature_names)
        df[self.label_name] = y_train
        self.model.fit(train_data=df, hyperparameters={self.clf_name:{}}, verbosity=0)
        self.feature_importances_ = self.feature_importance(df)

    def feature_importance(self, df):
        importances = []
        f_imp = self.model.feature_importance(df)
        for feature in self.feature_names:
            if feature in f_imp.importance.index.tolist():
                importances.append(abs(f_imp.importance.get(feature)))
            else:
                importances.append(0.0)
        return numpy.asarray(importances)

    def classifier_predict(self, x_test):
        df = pandas.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict(df, as_pandas=False)

    def classifier_predict_proba(self, x_test):
        df = pandas.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict_proba(df, as_pandas=False)


class FastAI(AutoGluonClassifier):
    """
    Wrapper for the gluon.FastAI algorithm
    """

    def __init__(self, label_name, metric="accuracy"):
        AutoGluonClassifier.__init__(self, label_name, "FASTAI", metric)

    def get_name(self):
        return "FastAI"


class GBM(AutoGluonClassifier):
    """
    Wrapper for the gluon.LightGBM algorithm
    """

    def __init__(self, label_name, metric="accuracy"):
        AutoGluonClassifier.__init__(self, label_name, "GBM", metric)

    def get_name(self):
        return "GBM"


class XGB(AutoGluonClassifier):
    """
    Wrapper for the gluon.XGB algorithmz
    """

    def __init__(self, label_name, metric="accuracy"):
        AutoGluonClassifier.__init__(self, label_name, "XGB", metric)

    def get_name(self):
        return "XGB"
