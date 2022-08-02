##################################################################################
# (c)  Copyright 2022 Hyojoon Kim
# All Rights Reserved 
# 
# email: joonk@gatech.edu
##################################################################################

###################################################################################
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.    
##################################################################################


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance
import forestci as fci
import scipy.stats

import tensorflow as tf

import argparse
import sys, os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import ml_library as ml_lib
import compare_auc_delong_xu

NUMBER_OF_ESTIMATORS = 200
NUMBER_OF_NN_RUNS = 20
NUMBER_OF_OTHER_RUNS = 20
NUMBER_OF_EPOCHS = 50
K_FOLD = 100

### Feature to LableName Mapping
feat_to_labelname_dict = {
    'BMI': 'BMI',
    'bmi_eg_25': 'BMI => 25',
    'pivot shift grade': 'Pivot shift grade',
    'pivot 2 groups': 'Pivot > 1',
    'mri_lat': 'Lateral tibial slope', 
    'age': 'Age', 
    'age_eg_30': 'Age => 30',
    'has_bone_contusion_ltp': 'LTP bone contusion', 
    'sex': 'Sex', 
    'has_segond_fx': 'Segond fx', 
    'has_bone_contusion_lfc': 'LFC bone contusion', 
    'Deep sulcus sign': 'Deep sulcus sign', 
    'meniscal_slope_lat': 'Lateral meniscal slope', 
    'has_mci_damage': 'MCL damage', 
    'has_bone_contusion_mfc': 'MFC bone contusion', 
    'Injury Mechanism': 'Injury mechanism', 
    'Ratio of LFC (1/2)': 'LFC ratio', 
    'inj_to_op_time_acute0_chronic1': 'Time from inj. to op. (acute/chronic)', 
    'S to S': 'S to S', 
    'has_bone_contusion_mtp': 'MTP bone contusion', 
    'mri_med': 'Medial tibial slope', 
    'alignment': 'Alignment (Varus/Valgus)', 
    'alignment.1': 'Alignment (3 Degree Varus)', 
    'meniscal_slope_med': 'Medial meniscal slope',
    'coronal alignment': 'Coronal alignment',
    'mri_has_ramp': 'MRI diagnosis',
    'Telos(side to side diff)': 'Telos side to side difference'
}


## Data curation 
def curate_data(data):

    # Rename columns
    data.rename(columns={data.columns[0]:"name", data.columns[1]:"regid", data.columns[2]:"sex", 
                         data.columns[3]:"age", data.columns[4]:"age_eg_30", data.columns[5]:"height", 
                         data.columns[6]:"weight", data.columns[8]:"bmi_eg_25",
                         data.columns[9]:"inj_to_op_time", data.columns[10]:"inj_to_op_time_acute0_chronic1",
                         data.columns[18]:"mri_med", data.columns[19]:"meniscal_slope_med", data.columns[20]:"mri_lat",
                         data.columns[21]:"meniscal_slope_lat", 
                         data.columns[24]:"has_bone_contusion_lfc", data.columns[25]:"has_bone_contusion_mfc",
                         data.columns[26]:"has_bone_contusion_ltp", data.columns[27]:"has_bone_contusion_mtp",
                         data.columns[28]:"has_mci_damage", data.columns[29]:"has_segond_fx",data.columns[41]:"mri_has_ramp",
                         data.columns[31]:"has_ramp"},\
                         inplace = True)

    # Change sex value: M => 0, F => 1.
    data.replace({'sex': r'^\s*M\s*$'}, {'sex': 0}, regex=True, inplace=True)
    data.replace({'sex': r'^\s*F\s*$'}, {'sex': 1}, regex=True, inplace=True)
 
    # swap +/- in "coronal alignment, HKA angle (varus - , valgus +)." Varus should be + and valgus should be -.
    data["coronal alignment"] = data["coronal alignment"]*-1


    # Get feature list and exclude items
    feature_index_list = data.columns
    exclude_list = ["name","regid"]
    feature_index_list = feature_index_list.drop(exclude_list)

    # The "alignment" value is based on "coronal alignment" value. (if negative, valgus (1). Else, varus (0)). 
    data["alignment"] = np.where(data['coronal alignment'] < 0, 1, 0)
 
    # The "alignment.1" value is based on "coronal alignment" value. (if > 6, 0. Else, 1). 
    data["alignment.1"] = np.where(data['coronal alignment'] >= 6, 0, 1)

    # Set "has_ramp" as label
    x = data[feature_index_list.drop("has_ramp")] # Features
    y = data['has_ramp']  # labels

    # Standardizing the features
#    x = StandardScaler().fit_transform(x)

    return data, y


## Define and set groups
def define_and_set_groups_2():
    list_of_fgrouplist = None

    age_group = ["age", "age_eg_30"]
    hw_bmi_group = ["BMI","bmi_eg_25"]
    time_to_inj_group = ["inj_to_op_time_acute0_chronic1"]
    pivot_group = ["pivot shift grade", "pivot 2 groups"]
#    alignment_group = ["coronal alignment", "alignment", "alignment.1"]
    alignment_group = ["alignment.1"]
    telos_group = ["Telos(side to side diff)", "S to S"]

    list_of_fgrouplist = [age_group, hw_bmi_group, time_to_inj_group, \
                         pivot_group, alignment_group,\
                         telos_group]

    return list_of_fgrouplist


## Try combination of features for neural network    
def try_combinations_nn(data, y, list_of_fgrouplist, nrun):

    fset_perf_map = {}

    x_list = []
    count = 1
    last = nrun*48

    for feat0 in list_of_fgrouplist[0]:
        for feat1 in list_of_fgrouplist[1]:
            for feat2 in list_of_fgrouplist[2]:
                for feat3 in list_of_fgrouplist[3]:
                    for feat4 in list_of_fgrouplist[4]:
                        for feat5 in list_of_fgrouplist[5]:
                                    
                            # Create feature list
                            x_list = [feat0,feat1,feat2,feat3,feat4,feat5]
                            x_list = x_list + ["sex", "has_mci_damage", "has_segond_fx", "Injury Mechanism", \
                                               "mri_med", "meniscal_slope_med", "mri_lat", "meniscal_slope_lat",\
                                               "has_bone_contusion_lfc", "has_bone_contusion_mfc", "has_bone_contusion_ltp", "has_bone_contusion_mtp",\
                                               "Deep sulcus sign","Ratio of LFC (1/2)", "mri_has_ramp"]

                            x = data[x_list] # Features

                            x = StandardScaler().fit_transform(x)
                
                            # Train test split
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test

                            # Run
                            list_of_results = []
                            results = {}
                            for i in range(nrun):

                                model = get_compiled_model()
                                model.fit(x_train, y_train, epochs=NUMBER_OF_EPOCHS, batch_size=5)
                               
                                y_proba = model.predict(x_test)[:]
                                y_classes = (model.predict(x_test) > 0.5).astype("int32")

                                confusion = metrics.confusion_matrix(y_test, y_classes)

                                tp = confusion[1, 1]
                                tn = confusion[0, 0]
                                fp = confusion[0, 1]
                                fn = confusion[1, 0]
                                
                                accuracy = metrics.accuracy_score(y_test, y_classes)
                                sensitivity = metrics.recall_score(y_test, y_classes)
                                specificity = tn / (tn + fp)
                                
                                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                                
                                results = {"model": model, "accuracy": accuracy, "specificity": specificity,\
                                       "sensitivity": sensitivity, "fpr_tpr": (fpr,tpr),\
                                        "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "x": x, "y": y}
                                list_of_results.append(results)

                                print("{} done out of {}.".format(count, last))
                                count+=1

                            fset_perf_map[";".join(x_list)] = list_of_results
    return fset_perf_map


## Try combination of features for RF and LR.    
def try_combinations(data, y, list_of_fgrouplist, nrun, classifier="rf"):

    fset_perf_map = {}

    x_list = []

    for feat0 in list_of_fgrouplist[0]:
        for feat1 in list_of_fgrouplist[1]:
            for feat2 in list_of_fgrouplist[2]:
                for feat3 in list_of_fgrouplist[3]:
                    for feat4 in list_of_fgrouplist[4]:
                        for feat5 in list_of_fgrouplist[5]:
                                    
                            # Create feature list
                            x_list = [feat0,feat1,feat2,feat3,feat4,feat5]
                            x_list = x_list + ["sex", "has_mci_damage", "Injury Mechanism", \
                                               "mri_med", "meniscal_slope_med",\
                                               "has_bone_contusion_mfc", "has_bone_contusion_mtp",\
                                               "Deep sulcus sign","Ratio of LFC (1/2)", "mri_has_ramp"]

                            x = data[x_list] # Features

                            x = StandardScaler().fit_transform(x)

                            # Train test split
                            # print(x_list)
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test

                            # Run
                            list_of_results = []
                            results = {}
                            for i in range(nrun):
                                model, accuracy, specificity, sensitivity, class_report = None, None, None, None, None
                                if classifier=="lr": 
                                    model, accuracy, specificity, sensitivity, class_report = \
                                        ml_lib.run_train_until_accuracy_atleast_lr(x_train, x_test, y_train, y_test, 0.0, x_list, do_print=1)
                                else: # default to random forest
                                    model, accuracy, specificity, sensitivity, class_report = \
                                        ml_lib.run_train_until_accuracy_atleast(x_train, x_test, y_train, y_test, 0.0, x_list, do_print=1,n_estimators=NUMBER_OF_ESTIMATORS)

                                results = {"model": model, "accuracy": accuracy, "specificity": specificity,\
                                           "sensitivity": sensitivity, "class_report": class_report,\
                                           "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "x": x, "y": y}
                                list_of_results.append(results)

                                y_proba = model.predict_proba(x_test)[:,1]
                                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                                results["fpr_tpr"] = (fpr,tpr)

                            fset_perf_map[";".join(x_list)] = list_of_results
    return fset_perf_map


def get_compiled_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(21, input_shape=(21,), activation='relu'),
      tf.keras.layers.Dense(5, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def get_model_by_roc(fset_perf_map, percentile, is_nn=False):

    fset_oneresult_dict = {}

    for fset in fset_perf_map:

        sort_by_roc = sorted(fset_perf_map[fset], key=lambda x: auc(x["fpr_tpr"][0], x["fpr_tpr"][1]), reverse=True)
        index = int(len(sort_by_roc)*percentile)
        if index>len(sort_by_roc)-1:
            index = len(sort_by_roc)-1
        result = sort_by_roc[index]

        fpr = result["fpr_tpr"][0]
        tpr = result["fpr_tpr"][1]
        n_roc_auc = auc(fpr, tpr)

        fset_oneresult_dict[fset] = (n_roc_auc,result)

    sort_fset_by_roc = sorted(fset_oneresult_dict.items(), key=lambda x: x[1][0], reverse=True)

    fset = sort_fset_by_roc[0][0]
    accuracy  = sort_fset_by_roc[0][1][0]
    result = sort_fset_by_roc[0][1][1]

    return fset, accuracy, result



def get_model_by_accuracy(fset_perf_map, percentile, is_nn=False):

    fset_oneresult_dict = {}

    for fset in fset_perf_map:

        sort_by_acc = sorted(fset_perf_map[fset], key=lambda x: x["accuracy"], reverse=True)
        index = int(len(sort_by_acc)*percentile)
        result = sort_by_acc[index]
        acc = result["accuracy"]

        fset_oneresult_dict[fset] = (acc,result)

            
    sort_fset_by_acc = sorted(fset_oneresult_dict.items(), key=lambda x: x[1][0], reverse=True)
    if is_nn:
        print(sort_fset_by_acc)

    fset = sort_fset_by_acc[0][0]
    accuracy  = sort_fset_by_acc[0][1][0]
    result = sort_fset_by_acc[0][1][1]

    return fset, accuracy, result


def get_max_model_by_accuracy(fset_perf_map):

    fset_max_dict = {}

    for fset in fset_perf_map:
        max_acc = 0;
        max_result = None
        for result in fset_perf_map[fset]:
            if max_acc < result["accuracy"]:
                max_acc = result["accuracy"]
                max_result = result

        fset_max_dict[fset] = (max_acc,max_result)

    sort_by_acc = sorted(fset_max_dict.items(), key=lambda x: x[1][0], reverse=True)

    fset = sort_by_acc[0][0]
    accuracy  = sort_by_acc[0][1][0]
    result = sort_by_acc[0][1][1]

    return fset, accuracy, result


def plot_roc_curve_jstyle(label, result, index):

    patterns = ['-', '--',':']
    colors = ['k', 'r','b']

    fpr = result["fpr_tpr"][0]
    tpr = result["fpr_tpr"][1]
    roc_auc = auc(fpr, tpr)
         
    plt.plot(fpr, tpr, patterns[index], label=label + ": " + str(roc_auc), color=colors[index])


def check_and_read(input_dir,efile,nn=False):
    data, y, list_of_fgrouplist  = None, None, None

    # check if processed data exists.
    d1_file = input_dir + "data.pkl"
    d2_file = input_dir + "y.pkl"

    if os.path.isfile(d1_file) and os.path.isfile(d2_file):
        with open(d1_file, "rb") as fd:
            data  = pickle.load(fd)
        with open(d2_file, "rb") as fd:
            y  = pickle.load(fd)
    else:
        data, y, list_of_fgrouplist = read_data_call(efile)
        # Save class_report
        fd = open(input_dir + "data.pkl", 'wb') 
        pickle.dump(data, fd) 
        fd.close()
        fd = open(input_dir + "y.pkl", 'wb') 
        pickle.dump(y, fd) 
        fd.close()

    if list_of_fgrouplist is None:
        if nn:
            list_of_fgrouplist = define_and_set_groups_2()
        else:
            list_of_fgrouplist = define_and_set_groups_2()

    return data, y, list_of_fgrouplist


def read_data_call(efile):

    print("Reading data...")

    # Read data
    df = ml_lib.read_excel_data(efile, 
                                "final_results_stats",
                                1, "A:AP", [2],361)
    data, y = curate_data(df)

    print(y)

    # Create groups of features
    list_of_fgrouplist =  define_and_set_groups_2()

    print("Reading data done.")
    return data, y, list_of_fgrouplist


def plot_partial_dependence_plot(model, x_train, fset, output_dir, filename, plot_title,fset_to_plot=None):
    fset_list = fset.split(';')
    features = []
    if fset_to_plot is None:
        features = [*range(len(fset_list))]
    else:
        features = fset_to_plot
    plot_partial_dependence(model, x_train, features, n_jobs=3, feature_names=fset_list)
    fig = plt.gcf()
    fig.suptitle(plot_title)
    fig.subplots_adjust(hspace=0.7)
    fig.savefig(output_dir + filename)


def rf_permutation_importance(model, x_test, y_test, fset, output_dir, filename):
    fset_list = fset.split(';')
    result = permutation_importance(model, x_test, y_test, n_repeats=5,random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    labels_list = []
    for s in sorted_idx:
        labels_list.append(feat_to_labelname_dict[fset_list[s]])
    print(labels_list)
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,vert=False, labels=labels_list)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
#    plt.show()
    fig.savefig(output_dir + filename)


def k_fold_cv(lr_model, rf_model, nn_model, result_best_lr, result_best_rf, result_best_nn, k):
    
    lr_prediction_error_list = []
    rf_prediction_error_list = []
    nn_prediction_error_list = []

    # split in K differnet ways
    for i in range(k):
        # new split
        
        # LR 
        x_train, x_test, y_train, y_test = train_test_split(result_best_lr["x"], result_best_lr["y"], test_size=0.3) # 70% training and 30% test
        lr_score = lr_model.score(x_test, y_test)
        lr_prediction_error = 1 - lr_score
        lr_prediction_error_list.append(lr_prediction_error)

        # RF
        x_train, x_test, y_train, y_test = train_test_split(result_best_rf["x"], result_best_rf["y"], test_size=0.3) # 70% training and 30% test
        rf_score = rf_model.score(x_test, y_test)
        rf_prediction_error = 1 - rf_score
        rf_prediction_error_list.append(rf_prediction_error)

        # NN
        x_train, x_test, y_train, y_test = train_test_split(result_best_nn["x"], result_best_nn["y"], test_size=0.3) # 70% training and 30% test
        nn_score = rf_score
        nn_prediction_error = rf_prediction_error
        nn_prediction_error_list.append(nn_prediction_error)

    return np.average(lr_prediction_error_list), np.average(rf_prediction_error_list), np.average(nn_prediction_error_list)


def main():

    parser = argparse.ArgumentParser(description='Script for running ML on RAMP ACL data')
    parser.add_argument('-i', dest='input_f', action='store', required=True,
                        help='Input directory where the files to load are')
    parser.add_argument('-o', dest='output_f', action='store', required=True,
                        help='Output directory where to store outputs')
    parser.add_argument('-f', dest='excel_f', action='store', required=True,
                        help='The data file in Excel format')

    # Parse
    args = parser.parse_args()

    # Check
    input_dir, output_dir = None, None
    if args.input_f and os.path.isdir(args.input_f) and os.path.isdir(args.output_f): 
        input_dir = ml_lib.check_directory_and_add_slash(args.input_f)
        output_dir = ml_lib.check_directory_and_add_slash(args.output_f)
    else:   # If none of above, something is wrong.
        print("Directories are incorrect. Maybe does not exist? Abort.\n")
        sys.exit(-1)

    if os.path.isfile(args.excel_f) is False:
        print("Given file is not a file or does not exist. Abort.\n")
        sys.exit(-1)
    
    efile = args.excel_f

    # setups
    fset_perf_map_lr, fset_perf_map_rf, fset_perf_map_nn = None, None, None
    lr_was_there, rf_was_there, nn_was_there = False, False, False
    data, y, list_of_fgrouplist = None, None, None

    fset_best_lr, fset_best_rf, fset_best_nn = None, None, None
    result_best_lr, result_best_rf, result_best_nn = None, None, None
    model_best_lr, model_best_rf, model_best_nn = None, None, None

    # NN
    rfile = input_dir + "fset_best_nn.pkl"
    if os.path.isfile(rfile):
        print("NN model exists. Just using that")
        model_best_nn = tf.keras.models.load_model(input_dir + 'nn.model')
        with open(input_dir + "fset_best_nn.pkl", "rb") as fd:
            fset_best_nn  = pickle.load(fd)
        with open(input_dir + "result_best_nn.pkl", "rb") as fd:
            result_best_nn  = pickle.load(fd)

        nn_was_there = True
    else:
        print("NN map does not exist. Running new.")
        n_roc_auc = 0.0
        ncount = 1
        data, y, list_of_fgrouplist = check_and_read(input_dir,efile,nn=True)
        while n_roc_auc < 0.50: 
            print("ROC: {}... NN trying again...{}th attempt".format(n_roc_auc,ncount))
            fset_perf_map_nn = try_combinations_nn(data,y, list_of_fgrouplist, NUMBER_OF_NN_RUNS)
            fset, accuracy, result = get_model_by_roc(fset_perf_map_nn, 0.95, is_nn=True)
            fpr = result["fpr_tpr"][0]
            tpr = result["fpr_tpr"][1]
            n_roc_auc = auc(fpr, tpr)
            ncount += 1

    # LR
    rfile = input_dir + "fset_perf_map_lr.pkl"
    if os.path.isfile(rfile):
        print("LR map exists. Just using that")
        model_best_lr = joblib.load(input_dir + "lg.model")
        with open(input_dir + "fset_best_lr.pkl", "rb") as fd:
            fset_best_lr  = pickle.load(fd)
        with open(input_dir + "result_best_lr.pkl", "rb") as fd:
            result_best_lr  = pickle.load(fd)
        lr_was_there = True
    else:
        print("LR map does not exist. Running new.")
        data, y, list_of_fgrouplist = check_and_read(input_dir,efile,nn=True)
        fset_perf_map_lr = try_combinations(data, y, list_of_fgrouplist, NUMBER_OF_OTHER_RUNS, "lr")

    # RF
    rfile = input_dir + "fset_perf_map_rf.pkl"
    if os.path.isfile(rfile):
        print("RF map exists. Just using that")
        model_best_rf = joblib.load(input_dir + "rf.model")
        with open(input_dir + "fset_best_rf.pkl", "rb") as fd:
            fset_best_rf  = pickle.load(fd)
        with open(input_dir + "result_best_rf.pkl", "rb") as fd:
            result_best_rf  = pickle.load(fd)
        rf_was_there = True
    else:
        print("RF map does not exist. Running new.")
        data, y, list_of_fgrouplist = check_and_read(input_dir,efile,nn=True)
        fset_perf_map_rf = try_combinations(data, y, list_of_fgrouplist, NUMBER_OF_OTHER_RUNS, "rf")

    # Initialize ROC value     
    roc = 0


    ## Start Analysis

    # NN
    if not nn_was_there:
        fset_best_nn, roc, result_best_nn = get_model_by_roc(fset_perf_map_nn, 0.95, is_nn=True)
        model_best_nn = result_best_nn["model"]

    print("\n")
    print("* NN: " + str(roc) + ": " + str(fset_best_nn))
    print("* NN. Accuracy: {}, specificity: {}, sensitivity: {}".format(result_best_nn["accuracy"], 
                                                                      result_best_nn["specificity"], 
                                                                      result_best_nn["sensitivity"]))
    if not nn_was_there: 
        ml_lib.save_model_and_fsetmap(result_best_nn["model"], input_dir + "nn.model", 
                                      fset_perf_map_nn, input_dir + "fset_perf_map_nn", 
                                      fset_best_nn, result_best_nn, input_dir, "nn", tensorflow_model = True)



    ## LR
    if not lr_was_there:
        fset_best_lr, roc, result_best_lr = get_model_by_roc(fset_perf_map_lr, 0.95)
        
    print("\n")
    print("* LR: " + str(roc) + ": " + str(fset_best_lr))
    print("* LR. Accuracy: {}, specificity: {}, sensitivity: {}".format(result_best_lr["accuracy"], 
                                                                      result_best_lr["specificity"], 
                                                                      result_best_lr["sensitivity"]))
    ll = fset_best_lr.split(';')
#    plot_partial_dependence_plot(result_best_lr["model"], result_best_lr["x_train"], 
#                                 "age;bmi_eg_25;inj_to_op_time_acute0_chronic1;pivot 2 groups;alignment.1;S to S;sex;has_mci_damage;has_segond_fx;Injury Mechanism;Medial tibial slope;Medial meniscal slope;mri_lat;meniscal_slope_lat;has_bone_contusion_lfc;has_bone_contusion_mfc;has_bone_contusion_ltp;has_bone_contusion_mtp;Deep sulcus sign;LFC ratio", 
#                                 output_dir, "lr_pdp.png", "Partial Dependence of Features in LR", [10,11,19])
    if not lr_was_there:
        ml_lib.save_model_and_fsetmap(result_best_lr["model"], input_dir + "lg.model", 
                                      fset_perf_map_lr, input_dir + "fset_perf_map_lr", 
                                      fset_best_lr, result_best_lr, input_dir, "lr")
    if model_best_lr is None:
        model_best_lr = result_best_lr["model"]


    ## RF
    if not rf_was_there: 
        fset_best_rf, roc, result_best_rf = get_model_by_roc(fset_perf_map_rf, 0.95)
    
    print("\n")
    print("* RF: " + str(roc) + ": " + str(fset_best_rf))
    print("* RF. Accuracy: {}, specificity: {}, sensitivity: {}".format(result_best_rf["accuracy"], 
                                                                      result_best_rf["specificity"], 
                                                                      result_best_rf["sensitivity"]))
    plt.figure()
#    plot_partial_dependence_plot(result_best_rf["model"], result_best_rf["x_train"], "age;BMI;inj_to_op_time_acute0_chronic1;pivot shift grade;alignment;S to S;sex;has_mci_damage;has_segond_fx;Injury Mechanism;Medial tibial slope;Medial meniscal slope;mri_lat;meniscal_slope_lat;has_bone_contusion_lfc;has_bone_contusion_mfc;has_bone_contusion_ltp;has_bone_contusion_mtp;Deep sulcus sign;LFC ratio", output_dir, "rf_pdp.png", "Partial Dependence of Features in RF",[10,11,19])
    if not rf_was_there: 
        ml_lib.save_model_and_fsetmap(result_best_rf["model"], input_dir + "rf.model", 
                                      fset_perf_map_rf, input_dir + "fset_perf_map_rf", 
                                      fset_best_rf, result_best_rf, input_dir, "rf")

    importances = result_best_rf["model"].feature_importances_
#    print(importances)

    rf_permutation_importance(result_best_rf["model"], result_best_rf["x_test"], 
                              result_best_rf["y_test"], fset_best_rf, output_dir, "perm_importance_rf.png")


    # Create new plot figure for ROC curve
    plt.figure()

    plot_roc_curve_jstyle("LogisticRegression", result_best_lr, 0)
    plot_roc_curve_jstyle("RandomForest", result_best_rf, 1)
    plot_roc_curve_jstyle("NeuralNetwork", result_best_nn, 2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(output_dir + 'roc_curve.png')
#    plt.show()          


    print("\n============= PRINT RESULTS =============")

    # LR
    alpha = .95
    y_pred = result_best_lr["model"].predict_proba(result_best_lr["x_test"])[:,1]
    y_true = result_best_lr["y_test"]
    aucvar, auc_cov = compare_auc_delong_xu.delong_roc_variance(y_true,y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q,loc=aucvar, scale=auc_std)
    ci[ci > 1] = 1
    print('LR AUC:', aucvar)
    print('LR AUC COV:', auc_cov)
    print('LR 95% AUC CI:', ci)
    print("\n")

    # RF
    alpha = .95
    y_pred = result_best_rf["model"].predict_proba(result_best_rf["x_test"])[:,1]
    y_true = result_best_rf["y_test"]
    aucvar, auc_cov = compare_auc_delong_xu.delong_roc_variance(y_true,y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q,loc=aucvar, scale=auc_std)
    ci[ci > 1] = 1
    print('RF AUC:', aucvar)
    print('RF AUC COV:', auc_cov)
    print('RF 95% AUC CI:', ci)
    print("\n")

    # NN
    alpha = .95
    y_pred = model_best_nn.predict(result_best_nn["x_test"])[:,0]
    y_true = result_best_nn["y_test"]
    aucvar, auc_cov = compare_auc_delong_xu.delong_roc_variance(y_true,y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q,loc=aucvar, scale=auc_std)
    ci[ci > 1] = 1
    print('NN AUC:', aucvar)
    print('NN AUC COV:', auc_cov)
    print('NN 95% AUC CI:', ci)
    print("\n")

    if model_best_rf is None:
        model_best_rf = result_best_rf["model"]

   # K-fold mean prediction error rate
    lr_avg_err_rate, rf_avg_err_rate, nn_avg_err_rate = k_fold_cv(model_best_lr, model_best_rf,  model_best_nn, result_best_lr, result_best_rf, result_best_nn, K_FOLD)

    print("LR Pred Error rate: {}. RF pred error rate: {}. NN pred error reate: {}".format(lr_avg_err_rate, rf_avg_err_rate, nn_avg_err_rate))
    print("============= END RESULTS =============\n")

if __name__ == '__main__':
    main()

