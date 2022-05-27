import glob
import time

import joblib
import numpy
import pandas
from sklearn import metrics
from negroni.utils.negroni_utils import get_name, load_tabular_dataset, get_meta_classifiers

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
    #     for model in get_meta_classifiers():
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
    col_names = None

    for dataset_name in datasets:
        x_tr, y_tr, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TRAIN"], label_name, 0)
        if big_x is None:
            big_x = x_tr
            col_names = x_tr.columns
        else:
            x_tr.columns = col_names
            big_x = pandas.concat([big_x, x_tr], ignore_index=True)
        if big_y is None:
            big_y = y_tr
        else:
            big_y = numpy.append(big_y, y_tr)
    big_x = big_x.fillna(0)
    big_x = big_x.to_numpy()

    models = []
    for clf in get_meta_classifiers():
        start = time.time()
        clf.fit(big_x, big_y)
        elapsed_train = (time.time() - start)
        models.append(clf)
        print(clf.feature_importances_)
        print("[" + get_name(clf) + "] trained in " + str(elapsed_train) + " ms, " +
              str(100*sum(big_y == 1)/len(big_y)) + "% of attacks")
        joblib.dump(clf, "models/" + get_name(clf) + ".joblib", compress=3)
        #pickle.dump(clf, open("models/" + get_name(clf) + ".pkl", "wb"))

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

    for dataset_to_avoid in datasets:
        print("\nAvoiding dataset " + dataset_to_avoid)

        big_x = None
        big_y = None
        col_names = None

        for dataset_name in datasets:
            if dataset_name != dataset_to_avoid:
                x_tr, y_tr, feature_list, att_rate = load_tabular_dataset(datasets[dataset_name]["TRAIN"], label_name, 0)
                if big_x is None:
                    big_x = x_tr
                    col_names = x_tr.columns
                else:
                    x_tr.columns = col_names
                    big_x = pandas.concat([big_x, x_tr], ignore_index=True)
                if big_y is None:
                    big_y = y_tr
                else:
                    big_y = numpy.append(big_y, y_tr)
        big_x = big_x.fillna(0)
        big_x = big_x.to_numpy()

        models = []
        for clf in get_meta_classifiers():
            start = time.time()
            clf.fit(big_x, big_y)
            elapsed_train = (time.time() - start)
            models.append(clf)
            print("[" + get_name(clf) + "] trained in " + str(elapsed_train) + " ms, " +
                  str(100*sum(big_y == 1)/len(big_y)) + "% of attacks")

        big_x = None
        big_y = None

        x_te, y_te, feature_list, att_rate = load_tabular_dataset(datasets[dataset_to_avoid]["TEST"], label_name, 0)

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

            print(dataset_to_avoid + " [AVOID] Accuracy/MCC/Rec = " + '{0:.4f}'.format(accuracy) + "/"
                  + '{0:.4f}'.format(mcc) + "/" + '{0:.4f}'.format(rec) + ", [" +
                  get_name(model) + "] time " + str(elapsed_test) + " ms")

            an_result = [elapsed_train, elapsed_test, tp, tn, fp, fn, accuracy, rec, mcc]
            to_print = dataset_to_avoid + "," + get_name(model) + ",avoid," + ",".join([str(x) for x in an_result])
            with open(scores_file, "a") as myfile:
                myfile.write(to_print + "\n")
