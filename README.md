# Predicting RAMP Lesions in ACL Deficient Knees
Predict the Presence of Ramp Lesions in ACL Deficient Knees using Machine Learning Techniques

## Prerequisites
  - Python3 (tested w/ v3.9.0)
  - pip3

## Setup and Running 
1. `$ git clone https://github.com/Medical-ML/ramp-acl.git`
1. `$ cd ramp-acl/`
1. `$ python3 -m venv venv`
1. `$ . ./venv/bin/activate`
1. `$ pip3 install -r requirements_ml.txt`
    - You might get an error about `ERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.` or some incompatibility issue with TensorFlow. But you can ignore these error for now.
1. If running for the first time without input and output directories made: 
    1. `$ mkdir input_dir;mkdir output_dir`
1. `$ python3 ramp_acl.py -i input_dir -o output_dir -f data.xlsx`
    - **NOTE**: The included `sample_data_format.xlsx` **DOES NOT** run correctly with this code since the Excel file does not contain enough valid data. Please just use it as a reference for figuring out the data format that works with this particular code. 
3. Results will be printed and figures will be saved in `output_dir`.  
