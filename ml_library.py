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
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os, sys


def read_excel_data(filepath, sheet_name, header=None, usecols=None, skiprows=None,nrows=None):
    df = pd.read_excel(filepath, sheet_name=sheet_name,header=header,usecols=usecols, skiprows=skiprows,nrows=nrows,engine='openpyxl')
    return df

def train_with_random_forest(x_train, x_test, y_train, y_test, n_esti=100):
    model = RandomForestClassifier(n_estimators=n_esti, oob_score=True)
#    model = RandomForestClassifier(n_estimators=n_esti, class_weight='balanced', oob_score=True)
#    model = RandomForestClassifier(n_estimators=n_esti, class_weight='balanced_subsample', oob_score=True)

    # Train the model 
    model.fit(x_train,y_train)

    return model


def train_with_lr(x_train, x_test, y_train, y_test):
    model = LogisticRegression()
#    model = LogisticRegression(class_weight='balanced')

    # Train the model 
    model.fit(x_train,y_train)

    return model


def train_with_svm(x_train, x_test, y_train, y_test):
#    model = svm.SVC(kernel='linear')
    model = svm.SVC(probability=True)

    # Train the model 
    model.fit(x_train,y_train)

    return model


def run_train_until_accuracy_atleast_svm(x_train, x_test, y_train, y_test, min_accuracy, x_list, do_print):
    model = None
    accuracy, specificity, sensitivity, class_report = None, None, None, None
    while 1:
        # Train model
        model = train_with_svm(x_train, x_test, y_train, y_test)

        # Predict and get accuracy
        accuracy, specificity, sensitivity, class_report = predict_with_model(model, x_test, y_test, x_list, do_print, classifier="svm")
 
        if float(accuracy) > min_accuracy:
            break

    return model, accuracy, specificity, sensitivity, class_report


def run_train_until_accuracy_atleast_lr(x_train, x_test, y_train, y_test, min_accuracy, x_list, do_print):
    model = None
    accuracy, specificity, sensitivity, class_report = None, None, None, None
    while 1:
        # Train model
        model = train_with_lr(x_train, x_test, y_train, y_test)

        # Predict and get accuracy
        accuracy, specificity, sensitivity, class_report = predict_with_model(model, x_test, y_test, x_list, do_print, classifier="lr")
 
        if float(accuracy) > min_accuracy:
            break

    return model, accuracy, specificity, sensitivity, class_report


def run_train_until_accuracy_atleast(x_train, x_test, y_train, y_test, min_accuracy, x_list, do_print, n_estimators=100):
    model = None
    accuracy, specificity, sensitivity, class_report = None, None, None, None
    while 1:
        # Train model
        model = train_with_random_forest(x_train, x_test, y_train, y_test, n_estimators)

        # Predict and get accuracy
        accuracy, specificity, sensitivity, class_report = predict_with_model(model, x_test, y_test, x_list, do_print)
 
        if float(accuracy) > min_accuracy:
            break

    return model, accuracy, specificity, sensitivity, class_report


def predict_with_model(model, x_test, y_test, x_list, do_print, classifier="rf"):

    y_pred = model.predict(x_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    importances = None
    if classifier == "rf":
        importances = model.feature_importances_

    if do_print: 
        print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred))
        print("\nSpecificity: ", class_report['0']['recall'])
        print("\nSensitivity: ", class_report['1']['recall'])

        print("\n")
        print(classification_report(y_test, y_pred))
        
        if classifier == "rf":
            print('Feature Importance:\n')
            for x,i in zip(x_list, importances):
                print(str(i)+":", x)
            
        print("\n")

    accuracy = metrics.accuracy_score(y_test, y_pred)
    specificity = class_report['0']['recall']
    sensitivity =  class_report['1']['recall']

    return accuracy, specificity, sensitivity, class_report


def save_model_and_fsetmap(model, model_name, fset_map, fset_map_name, fset, result, input_dir, model_type, tensorflow_model=False):

    this_result = result

    # Save model
    if tensorflow_model: 
        model.save(model_name)
        this_result["model"] = None
    else:
        joblib.dump(model, model_name)

        # Save class_report
        fd = open(fset_map_name+".pkl", 'wb') 
        pickle.dump(fset_map, fd) 
        fd.close()

    with open(input_dir + "fset_best_" + model_type + ".pkl", 'wb') as fd:
        pickle.dump(fset, fd) 

    with open(input_dir + "result_best_" + model_type + ".pkl", 'wb') as fd:
        pickle.dump(this_result, fd) 
    

def save_model_and_report_with_pickle(model, class_report, model_name):

    # Save model
    joblib.dump(model, model_name)

    # Save class_report
    fd = open('class_report.pkl', 'wb') 
    pickle.dump(class_report, fd) 
    fd.close()


def check_directory_and_add_slash(path):
    return_path = path
  
    # No path given  
    if return_path == None:
        print ("None is given as path. Check given parameter.")
        return ''
      
    # Path is not a directory, or does not exist.  
    if os.path.isdir(return_path) is False:
        print ("Path is not a directory, or does not exist.: '%s'. Abort" % (return_path))
        return ''
  
    # Add slash if not there.  
    if return_path[-1] != '/':
        return_path = return_path + '/'
  
    # Return
    return return_path

