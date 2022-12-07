import json
import os

import pandas as pd

import th_code.time_series_custom_extraction as tsce
import th_code.time_series_resample as tsr
import th_code.lstm as lstm

def read_metrics():
    path = os.path.dirname(__file__)
    new_path = os.path.relpath('./conf/', path)
    with open(new_path + '/metrics.json') as file:
        metrics = json.load(file)
        return [metric.replace('/', '_') for metric in metrics]

metrics = read_metrics()  # ['Axis_X_aaLoad', 'Axis_Y_aaLoad']

file = '1ab2f9dd-62ff-4433-8d88-605744403ab2.xes.yaml'
# file = 'test.xes.yaml'

filename = file.split('.')[0]
path = 'uploads/' + filename + '/single_runs/'

# tsce.extract(metrics, 'uploads/logs', path, file, filename)

print(read_metrics())



def write_csv(df, path2, file):
    print("write resampled " + file)
    df.to_csv(path2 + file, header=False)

path = 'uploads/' +  filename + '/single_runs/'
path2 = 'uploads/' +  filename + '/resampled/'


# tsr.resample(path, path2, os.listdir(path))

path = 'uploads/' + filename + '/resampled/'
path2 = 'uploads/' + filename + '/models/lstm'
path3 = 'uploads/' + filename + '/generated/'

files = os.listdir(path)

for file in files:

    lstm.train_models2(path, path2, path3, file, filename)