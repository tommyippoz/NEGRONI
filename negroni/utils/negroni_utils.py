import numpy
import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.suod import SUOD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from negroni.classifiers.NEGRONILearner import NEGRONILearner
from negroni.classifiers.NEGRONIWrapper import NEGRONIWrapper
from negroni.connectors.AutoGluonLearner import FastAI
from negroni.connectors.PYODLearner import PYODLearner
from negroni.ensembles.StackingLearner import StackingLearner


def get_name(learner):
    if isinstance(learner, NEGRONILearner):
        return learner.get_name()
    else:
        return learner.__class__.__name__


def load_tabular_dataset(dataset_name, label_name, normal_tag, limit=np.nan):
    """
    Method to process an input dataset as CSV
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)

    # Testing Purposes
    if (np.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == normal_tag]
    att_rate = 1 - len(normal_frame.index) / len(df.index)

    # Label encoding with integers
    df[label_name] = np.where(df[label_name] == normal_tag, 0, 1)
    y = np.asarray(df[label_name])
    x = df.select_dtypes(exclude=['object'])
    x = x.drop(columns=[label_name])
    feature_list = x.columns

    print("Dataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(len(normal_frame.index)) +
          " normal, " + str(len(feature_list)) + " features, " + str(100*att_rate) + "% of attacks")

    return x, y, feature_list, att_rate


def unsupervised_classifiers(outliers_fraction):
    class_list = []

    if outliers_fraction > 0.5:
        outliers_fraction = 0.5

    class_list.append(PYODLearner(COPOD(contamination=outliers_fraction)))
    class_list.append(PYODLearner(ABOD(contamination=outliers_fraction, method="fast")))
    class_list.append(PYODLearner(HBOS(contamination=outliers_fraction)))
    class_list.append(PYODLearner(MCD(contamination=outliers_fraction)))
    class_list.append(PYODLearner(PCA(contamination=outliers_fraction, weighted=True)))
    class_list.append(PYODLearner(ECOD(contamination=outliers_fraction)))
    class_list.append(PYODLearner(LOF(contamination=outliers_fraction, n_jobs=-1)))
    class_list.append(PYODLearner(CBLOF(contamination=outliers_fraction)))
    class_list.append(PYODLearner(KNN(contamination=outliers_fraction)))
    class_list.append(PYODLearner(IForest(contamination=outliers_fraction)))
    class_list.append(PYODLearner(SUOD(contamination=outliers_fraction, base_estimators=[COPOD(), PCA(), CBLOF()])))

    return class_list


def supervised_classifiers(label_name):
    return [NEGRONIWrapper(GaussianNB()),
            NEGRONIWrapper(LinearDiscriminantAnalysis()),
            NEGRONIWrapper(DecisionTreeClassifier()),
            NEGRONIWrapper(LogisticRegression(random_state=0)),
            NEGRONIWrapper(KNeighborsClassifier(9)),
            NEGRONIWrapper(FastAI(label_name=label_name)),
            NEGRONIWrapper(AdaBoostClassifier(n_estimators=10))]


def get_base_classifiers(att_rate, label_name):
    list = unsupervised_classifiers(att_rate)
    list.extend(supervised_classifiers(label_name))
    return list


def get_meta_classifiers():
    return [XGBClassifier(eval_metric="logloss"), RandomForestClassifier()]


def get_stackers(att_rate, label_name):
    att_rate = att_rate if att_rate < 0.5 else 0.5
    list = [StackingLearner(base_level_learners=get_base_classifiers(att_rate, label_name),
                            meta_level_learner=RandomForestClassifier(),
                            use_training=False, store_data=True, verbose=True)]
    return list


def sample_weight(weights, samples_n):
    indexes = numpy.random.choice(len(weights), samples_n, replace=False, p=weights)
    return indexes
