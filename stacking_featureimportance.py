import glob
import pickle
import time

import joblib
import numpy
import numpy as np
import pandas
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
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from negroni.connectors.AutoGluonLearner import FastAI
from negroni.connectors.PYODLearner import PYODLearner
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


def get_classifiers():
    return [XGBClassifier(eval_metric="logloss"), RandomForestClassifier()]


if __name__ == '__main__':

    label_name = "bin_label"
    input_dir = "stacking_datasets"
    scores_file = "output/feature_importances.csv"

    datasets = {}

    for file_path in glob.glob(input_dir + "/*.csv"):

        if "/" in file_path:
            csv_file = file_path.split("/")[-1]
        elif "\\" in file_path:
            csv_file = file_path.split("\\")[-1]
        csv_file = csv_file.replace(".csv", "").replace("StackingData_", "")

        if "TRAIN" in csv_file:
            dataset_name = csv_file.replace("_TRAIN", "")
            is_train = True
        elif "TEST" in csv_file:
            dataset_name = csv_file.replace("_TEST", "")
            is_train = False
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name]["TRAIN" if is_train else "TEST"] = file_path

    first_compute = True
    for dataset_name in datasets:

        x_tr, y_tr, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TRAIN"], label_name, 0)
        x_te, y_te, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TEST"], label_name, 0)

        if first_compute:
            # Setup Output File
            with open(scores_file, "w") as myfile:
                # Print Header
                myfile.write(
                    "datasetName,classifierName," + ",".join([str(f) for f in feature_list]) + "\n")
            first_compute = False

        for model in get_classifiers():

            # Train
            start = time.time()
            model.fit(x_tr, y_tr)
            elapsed_train = (time.time() - start)

            f_imp = ",".join([str(x) for x in model.feature_importances_])
            to_print = dataset_name + "," + get_name(model) + "," + f_imp
            with open(scores_file, "a") as myfile:
                myfile.write(to_print + "\n")

            model = None
