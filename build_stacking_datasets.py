import configparser
import glob
import time

from sklearn import metrics

from sklearn.model_selection import train_test_split

from negroni.utils.negroni_utils import get_name, load_tabular_dataset, get_stackers

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

        for model in get_stackers(att_rate, label_name):

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
