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


def get_base_level_classifiers(outliers_fraction):
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


def get_meta_level_classifiers():
    return [RandomForestClassifier()]


def get_classifiers_with_stacking(att_rate):
    list = []
    for clf in get_meta_level_classifiers():
        list.extend([StackingLearner(base_level_learners=get_base_level_classifiers(att_rate if att_rate < 0.5 else 0.5),
                                     meta_level_learner=clf, use_training=False, verbose=True)])
    return list


if __name__ == '__main__':

    label_name = "bin_label"
    input_dir = "stacking_datasets"
    scores_file = "output/stacker.csv"

    # Setup Output File
    with open(scores_file, "w") as myfile:
        # Print Header
        myfile.write(
            "datasetName,classifierName,trainTime,testTime,tp,tn,fp,fn,acc,rec,mcc\n")

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

    # for dataset_name in datasets:
    #
    #     x_tr, y_tr, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TRAIN"], label_name, 0)
    #     x_te, y_te, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TEST"], label_name, 0)
    #
    #     for model in get_meta_level_classifiers():
    #
    #         # Train
    #         start = time.time()
    #         model.fit(x_tr, y_tr)
    #         elapsed_train = (time.time() - start)
    #
    #         # Scoring Test Confusion Matrix
    #         start = time.time()
    #         y_pred = model.predict(x_te)
    #         elapsed_test = (time.time() - start)
    #
    #         tn, fp, fn, tp = metrics.confusion_matrix(y_te, y_pred).ravel()
    #         accuracy = metrics.accuracy_score(y_te, y_pred)
    #         mcc = abs(metrics.matthews_corrcoef(y_te, y_pred))
    #         if accuracy < 0.5:
    #             accuracy = 1.0 - accuracy
    #             tp, fn = fn, tp
    #             tn, fp = fp, tn
    #         rec = tp / (tp + fn)
    #
    #         print(dataset_name + " Accuracy/MCC/Rec = " + '{0:.4f}'.format(accuracy) + "/" + '{0:.4f}'.format(mcc) + "/" +
    #               '{0:.4f}'.format(rec) + ", [" + get_name(model) + "] time " + str(elapsed_train) + " ms")
    #
    #         an_result = [elapsed_train, elapsed_test, tp, tn, fp, fn, accuracy, rec, mcc]
    #         to_print = dataset_name + "," + get_name(model) + ",regtrain," + ",".join([str(x) for x in an_result])
    #         with open(scores_file, "a") as myfile:
    #             myfile.write(to_print + "\n")
    #
    #         model = None

    big_x = None
    big_y = None

    for dataset_name in datasets:
        x_tr, y_tr, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TRAIN"], label_name, 0)
        if big_x is None:
            big_x = x_tr
        else:
            big_x = pandas.concat([big_x, x_tr], ignore_index=True)
        if big_y is None:
            big_y = y_tr
        else:
            big_y = numpy.append(big_y, y_tr)
    big_x = big_x.fillna(0)
    big_x = big_x.to_numpy()

    models = []
    for clf in get_meta_level_classifiers():
        start = time.time()
        clf.fit(big_x, big_y)
        elapsed_train = (time.time() - start)
        models.append(clf)
        print("[" + get_name(clf) + "] trained in " + str(elapsed_train) + " ms, " +
              str(100*sum(big_y == 1)/len(big_y)) + "% of attacks")
        joblib.dump(clf, "models/" + get_name(clf) + ".joblib")
        pickle.dump(clf, open("models/" + get_name(clf) + ".pkl", "wb"))

    big_x = None
    big_y = None

    for dataset_name in datasets:

        x_te, y_te, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TEST"], label_name, 0)

        for model in models:
            start = time.time()
            y_pred = model.predict(x_te)
            elapsed_test = (time.time() - start)

            tn, fp, fn, tp = metrics.confusion_matrix(y_te, y_pred).ravel()
            accuracy = metrics.accuracy_score(y_te, y_pred)
            mcc = abs(metrics.matthews_corrcoef(y_te, y_pred))
            if accuracy < 0.5:
                accuracy = 1.0 - accuracy
                tp, fn = fn, tp
                tn, fp = fp, tn
            rec = tp / (tp + fn)

            print(dataset_name + " [BIGTRAIN] Accuracy/MCC/Rec = " + '{0:.4f}'.format(accuracy) + "/"
                  + '{0:.4f}'.format(mcc) + "/" + '{0:.4f}'.format(rec) + ", [" +
                  get_name(model) + "] time " + str(elapsed_test) + " ms")

            an_result = [elapsed_train, elapsed_test, tp, tn, fp, fn, accuracy, rec, mcc]
            to_print = dataset_name + "," + get_name(model) + ",bigtrain," + ",".join([str(x) for x in an_result])
            with open(scores_file, "a") as myfile:
                myfile.write(to_print + "\n")





