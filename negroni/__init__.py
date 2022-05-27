import configparser
import glob
import pickle
import time

import joblib
import numpy as np
import pandas as pd
import xgboost
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from negroni.classifiers.NEGRONIWrapper import NEGRONIWrapper
from negroni.connectors.AutoGluonLearner import FastAI
from negroni.connectors.PYODLearner import PYODLearner
from negroni.custom.SupUnsupStacker import SupUnsupStacker
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


def mixed_classifiers(att_rate, label_name):
    list = unsupervised_classifiers(att_rate)
    list.extend([NEGRONIWrapper(GaussianNB()),
                 NEGRONIWrapper(LinearDiscriminantAnalysis()),
                 NEGRONIWrapper(DecisionTreeClassifier()),
                 NEGRONIWrapper(LogisticRegression(random_state=0)),
                 NEGRONIWrapper(KNeighborsClassifier(9)),
                 NEGRONIWrapper(FastAI(label_name=label_name)),
                 NEGRONIWrapper(AdaBoostClassifier(n_estimators=10))])
    return list


def get_classifiers(att_rate):
    ratio = att_rate if att_rate < 0.5 else 0.5
    return [PYODLearner(COPOD(contamination=ratio)),
            PYODLearner(MCD(contamination=ratio)),
            PYODLearner(PCA(contamination=ratio)),
            PYODLearner(CBLOF(contamination=ratio)),
            PYODLearner(IForest(contamination=ratio)),
            BaggingLearner(PYODLearner(COPOD(contamination=ratio)))]


def get_classifiers_with_stacking(att_rate, label_name):
    att_rate = att_rate if att_rate < 0.5 else 0.5
    #big_model = joblib.load("../models/RandomForestClassifier.joblib")
    list = [StackingLearner(base_level_learners=mixed_classifiers(att_rate, label_name),
                            meta_level_learner=RandomForestClassifier(),
                            use_training=False, store_data=True, verbose=True)]
     #       SupUnsupStacker(outliers_fraction=att_rate, meta_level_model=big_model, verbose=True)]
    return list


if __name__ == '__main__':

    # Load configuration parameters
    config = configparser.ConfigParser()
    config.read('negroni.config')

    label_name = config['input']['LABEL_NAME']
    normal_tag = config['input']['NORMAL_TAG']
    input_dir = config['input']['DATASETS_DIR']
    tvs = float(config['input']['TRAIN_VALIDATION_SPLIT'])
    scores_file = config['output']['SCORES_FILENAME']
    output_dir = config['output']['OUTPUT_DIR']

    # Setup Output File
    with open(scores_file, "w") as myfile:
        # Print Header
        myfile.write(
            "datasetName,classifierName,trainTime,testTime,tp,tn,fp,fn,acc,rec,mcc\n")

    for csv_file in glob.glob(input_dir + "/*.csv"):

        # Loading tabular dataset
        x, y, feature_list, att_rate = load_tabular_dataset(csv_file, label_name, normal_tag)
        if "/" in csv_file:
            csv_file = csv_file.split("/")[-1].replace(".csv", "")
        elif "\\" in csv_file:
            csv_file = csv_file.split("\\")[-1].replace(".csv", "")

        # Partitioning Train/Test split
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=(1 - tvs))

        for model in get_classifiers_with_stacking(att_rate, label_name):

            # Train
            start = time.time()
            model.fit(x_tr, y_tr)
            elapsed_train = (time.time() - start)

            # Scoring Test Confusion Matrix
            start = time.time()
            y_pred = model.predict(x_te)
            elapsed_test = (time.time() - start)

            tr_df = model.get_stacking_data()
            tr_df.to_csv(output_dir + "/StackingAll_" + csv_file + "_TRAIN.csv", index=False)

            te_df = model.get_stacking_test()
            te_df["bin_label"] = y_te
            te_df.to_csv(output_dir + "/StackingAll_" + csv_file + "_TEST.csv", index=False)

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

            an_result = [elapsed_train, elapsed_test, tp, tn, fp, fn, accuracy, rec, mcc]
            to_print = csv_file + "," + get_name(model) + "," + ",".join([str(x) for x in an_result])
            with open(scores_file, "a") as myfile:
                myfile.write(to_print + "\n")

            model = None
