import configparser
import glob
import time

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from negroni.classifiers.PYODLearner import PYODLearner
from negroni.ensembles.BaggingLearner import BaggingLearner
from negroni.ensembles.BoostingLearner import BoostingLearner
from negroni.ensembles.StackingLearner import StackingLearner
from negroni.utils.negroni_utils import get_name


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
    normal_frame = df.loc[df[label_name] == "normal"]
    att_rate = 1 - len(normal_frame.index) / len(df.index)
    print("Dataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(len(normal_frame.index)) +
          " normal")

    # Label encoding with integers
    df[label_name] = np.where(df[label_name] == normal_tag, 0, 1)
    y = np.asarray(df[label_name])
    x = df.select_dtypes(exclude=['object'])
    x = x.drop(columns=[label_name])
    feature_list = x.columns

    return x, y, feature_list, att_rate


def get_classifiers(att_rate):
    ratio = att_rate if att_rate < 0.5 else 0.5
    return [COPOD(contamination=ratio),
            BaggingLearner(PYODLearner(COPOD(contamination=ratio))),
            BoostingLearner(PYODLearner(COPOD(contamination=ratio)))
            ]
            # GaussianNB(),
            # BaggingLearner(GaussianNB()),
            # BoostingLearner(GaussianNB()),
            # LinearDiscriminantAnalysis(),
            # BaggingLearner(LinearDiscriminantAnalysis()),
            # BoostingLearner(LinearDiscriminantAnalysis())]


def get_classifiers_with_stacking():
    list = get_classifiers()
    list.extend([XGBClassifier(),
                 StackingLearner(base_level_learners=get_classifiers(), meta_level_learner=XGBClassifier(), use_training=False),
                 StackingLearner(base_level_learners=get_classifiers(), meta_level_learner=XGBClassifier(), use_training=True)])
    return list


if __name__ == '__main__':

    # Load configuration parameters
    config = configparser.ConfigParser()
    config.read('negroni.config')

    label_name = config['input']['LABEL_NAME']
    normal_tag = config['input']['NORMAL_TAG']
    input_dir = config['input']['DATASETS_DIR']
    tvs = float(config['input']['TRAIN_VALIDATION_SPLIT'])

    for csv_file in glob.glob(input_dir + "/*.csv"):

        # Loading tabular dataset
        x, y, feature_list, att_rate = load_tabular_dataset(csv_file, label_name, normal_tag)
        if "/" in csv_file:
            csv_file = csv_file.split("/")[-1]
        elif "\\" in csv_file:
            csv_file = csv_file.split("\\")[-1]

        # Partitioning Train/Test split
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=(1-tvs))

        for model in get_classifiers(att_rate):

            # Train
            start = time.time()
            model.fit(x_tr, y_tr)
            elapsed_train = (time.time() - start) / len(y_tr)

            # Scoring Test Confusion Matrix
            start = time.time()
            y_pred = model.predict(x_te)
            elapsed_test = (time.time() - start) / len(y_te)
            tn, fp, fn, tp = metrics.confusion_matrix(y_te, y_pred).ravel()
            accuracy = metrics.accuracy_score(y_te, y_pred)
            mcc = abs(metrics.matthews_corrcoef(y_te, y_pred))
            if accuracy < 0.5:
                accuracy = 1.0 - accuracy
                tp, fn = fn, tp
                tn, fp = fp, tn
            rec = tp / (tp + fn)

            print("Accuracy/MCC/Rec = " + '{0:.4f}'.format(accuracy) + "/" + '{0:.4f}'.format(mcc) + "/" +
                  '{0:.4f}'.format(rec) + ", [" + get_name(model) + "] time " + str(elapsed_train) + " ms")

            model = None