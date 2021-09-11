#!/usr/bin/env python
"""
This code loads preprocessed data, creates train and test sets,
load the specified pretrained model and evaluates in on the test set.

- It requires to have preprocessed data in 'clean_data'
"""
import pickle
import os.path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import cv2
import keras
import kapre

from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_fscore_support, matthews_corrcoef
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras.layers import LSTM, Convolution1D, Activation, AveragePooling1D
from keras.utils import np_utils

parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Evaluate the given physiological model, options:stresnet, fcn or resnet", type=str)
args = parser.parse_args()
if args.model_name=="stresnet" or args.model_name=="fcn" or args.model_name=="resnet": 
    model_name = args.model_name
else:
    raise NameError("{} is not supported".format(args.model_name))


show_debug = False
cleaned_data_dir = r"/data"
list_ = []
relevant_features = ["Palm.EDA", "Heart.Rate", "Breathing.Rate", "Perinasal.Perspiration", "Acceleration", "Steering",
                     "Brake", "Speed"]

for subdir, dirs, files in os.walk(cleaned_data_dir):
    for file in files:
        if file.startswith('clean__'):
            csv_path = os.path.join(subdir, file)
            df = pd.read_csv(csv_path, index_col=None, header=0, sep=',')
            list_.append(df)


def concat_ordered_columns(frames):
    columns_ordered = []
    for frame in frames:
        columns_ordered.extend(x for x in frame.columns if x not in columns_ordered)
    final_df = pd.concat(frames)
    return final_df[columns_ordered]


df_full = concat_ordered_columns(list_)

if show_debug is True:
    print(df_full.shape)

indsNaN = pd.isnull(df_full).any(1).to_numpy().nonzero()[0]

if show_debug is True:
    print(indsNaN)

# Drop NaN values
df_full = df_full.dropna()

# All Data
fullData = df_full[relevant_features]
y_labels = df_full.iloc[:, 23:31]

# drop Disgust label and it's not relevant for driving and too similar to neutral
y_labels = y_labels.drop(columns=['Disgust'])

# To have categorical labels, take the max of each row in the dataframe as the ground truth
maxLabels = y_labels.idxmax(axis=1)

maxLabels[maxLabels == 'Fear'] = '0'
maxLabels[maxLabels == 'Sad'] = '1'
maxLabels[maxLabels == 'Contempt'] = '2'
maxLabels[maxLabels == 'Anger'] = '3'
maxLabels[maxLabels == 'Surprise'] = '4'
maxLabels[maxLabels == 'Joy'] = '4'
maxLabels[maxLabels == 'Neutral'] = '4'
maxLabels = maxLabels.fillna('4')

if show_debug is True:
    print(len(maxLabels[maxLabels == '0']))
    print(len(maxLabels[maxLabels == '1']))
    print(len(maxLabels[maxLabels == '2']))
    print(len(maxLabels[maxLabels == '3']))
    print(len(maxLabels[maxLabels == '4']))

# Encode labels to Integer for One-hot encoding
le = LabelEncoder()
y_labels['Max'] = le.fit_transform(maxLabels)

# Normalize training and test data
trainingValues = fullData.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
trainingValues_scaled = min_max_scaler.fit_transform(trainingValues)
fullData = pd.DataFrame(trainingValues_scaled, columns=fullData.columns)

# Reconcatenate the y_labels to training data after normalization
fullData = fullData.reset_index(drop=True)
y_labels = y_labels.reset_index(drop=True)
y_original = y_labels.to_numpy()

# Can change to full labels if necessary, set to only the encoded label values currently.
fullData = pd.concat([fullData, y_labels], axis=1)

# Drop positive nd neutral emotions as we only focus on negative emotions
fullData = fullData[fullData["Max"] != 4]
if show_debug is True:
    print(fullData.shape)
y_labels = fullData.iloc[:, 8:16]

# Create Training and Test set
x_train, x_test, y_train, y_test = train_test_split(fullData[relevant_features], y_labels, test_size=0.2,
                                                    random_state=1, shuffle=False)

physiological_features = ["Palm.EDA", "Heart.Rate", "Breathing.Rate", "Perinasal.Perspiration"]


def create_list(x_train, x_test):
    x_train_array = []
    x_test_array = []
    x_train_array.append(x_train)
    x_test_array.append(x_test)
    return x_train_array, x_test_array


y_train_array = []
y_test_array = []
y_train_array.append(y_train)
y_test_array.append(y_test)

x_train_physio, x_test_physio = create_list(x_train[physiological_features], x_test[physiological_features])


def create_segments_and_labels(df, time_steps, step, labeldf, isBehavioral):
    """
    This function receives a dataframe and returns the reshaped segments
    of the assorted features
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # list number of features to be extracted, can be changed depending on what features are desired to be extracted
    N_FEATURES = 4

    segments = []
    palm = []
    hr = []
    br = []
    per = []
    labels = []
    for j in range(0, len(df)):
        for i in range(1, len(df[j]) - time_steps, step):
            if isBehavioral is True:
                steering = df[j]['Steering'].values[i: i + time_steps]
                acceleration = df[j]['Acceleration'].values[i: i + time_steps]
                speed = df[j]['Speed'].values[i: i + time_steps]
                brake = df[j]['Brake'].values[i: i + time_steps]
                segments.append([steering, acceleration, speed, brake])
            else:
                palmEDA = df[j]['Palm.EDA'].values[i: i + time_steps]
                heartRate = df[j]['Heart.Rate'].values[i: i + time_steps]
                breathingRate = df[j]['Breathing.Rate'].values[i: i + time_steps]
                perinasalPerspiration = df[j]['Perinasal.Perspiration'].values[i: i + time_steps]
                palm.append([palmEDA])
                hr.append([heartRate])
                br.append([breathingRate])
                per.append([perinasalPerspiration])

            maxLabel = labeldf[j]['Max'].values[i: i + time_steps]
            labels.append(maxLabel)

        if isBehavioral is False:
            palm_r = np.asarray(palm, dtype=np.float32).reshape(-1, time_steps, 1)
            hr_r = np.asarray(hr, dtype=np.float32).reshape(-1, time_steps, 1)
            br_r = np.asarray(br, dtype=np.float32).reshape(-1, time_steps, 1)
            per_r = np.asarray(per, dtype=np.float32).reshape(-1, time_steps, 1)
            list_f = []
            list_f.append(palm_r)
            list_f.append(hr_r)
            list_f.append(br_r)
            list_f.append(per_r)
            labels = np.asarray(labels)

            return list_f, labels

        # Bring the segments into a better shape
        print(len(segments[0]))
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)

        return reshaped_segments, labels


# Variables for input to CNN
class_names = ['Fear', 'Sad', 'Contempt', 'Anger']
TIME_PERIODS = 30
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 1

x_train_physio, y_train = create_segments_and_labels(x_train_physio,
                                                     TIME_PERIODS,
                                                     STEP_DISTANCE,
                                                     y_train_array,
                                                     False
                                                     )

x_test_physio, y_test = create_segments_and_labels(x_test_physio,
                                                   TIME_PERIODS,
                                                   STEP_DISTANCE,
                                                   y_test_array,
                                                   False
                                                   )

# Can uncomment if y is a 2D vector
y_train = y_train[:, 0]
y_test = y_test[:, 0]

if show_debug:
    print(np.shape(x_train_physio))
    print(np.shape(x_train_physio[0]))
    print(x_train.shape[0], 'training samples')
    print(np.shape(y_train))

# Set input & output dimensions
num_time_periods, num_sensors = x_train_physio[0].shape[1], x_train_physio[0].shape[2]
# labels to train on
num_classes = 4
input_shape = (num_time_periods * num_sensors)

if show_debug is True:
    print('x_train shape:', x_train_physio[0].shape)
    print('input_shape:', input_shape)

# Convert type for Keras otherwise Keras cannot process the data
x_train_physio = [x.astype("float32") for x in x_train_physio]
x_test_physio = [x.astype("float32") for x in x_test_physio]

# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print('New y_train shape: ', y_train.shape)
print('New y_test shape: ', y_test.shape)


# Normalize training and test data
min_max_scaler = preprocessing.MinMaxScaler()
combined_tensor = x_train_physio
combined_tensor_test = x_test_physio

# Add class weights because the model is extremely unbalanced
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

model = load_model(r'./models/physiological_CNN/{}.h5'.format(model_name), custom_objects={'Spectrogram':kapre.time_frequency.Spectrogram})

def create_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    cmn = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cmn,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True,
                fmt=".2f")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('./confusion_matrix_{}.png'.format(model_name))

with session.as_default():
    y_pred_test = model.predict(combined_tensor_test)

# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

create_confusion_matrix(max_y_test, max_y_pred_test)

print("\n--- Classification report for test data ---\n")
print(classification_report(max_y_test, max_y_pred_test))


def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T


matthewCorrelation = matthews_corrcoef(max_y_test, max_y_pred_test)

print(matthewCorrelation)

df_class_report = pandas_classification_report(max_y_test, max_y_pred_test)
df_class_report.to_csv('./classification_report_{}.csv'.format(model_name), sep=',')

