#!/usr/bin/env python
"""
This code trains a Convolutional Neural Network on the synced data.
Directly predicts emotional status from only behavioral data.

- It requires to have preprocessed data in 'clean_data'
"""
import os.path
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import keras

from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report, precision_recall_fscore_support, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Convolution1D, Activation, AveragePooling1D
from keras.utils import np_utils

parser = argparse.ArgumentParser()
parser.add_argument("train", help="Train or evaluate a pretrained model, options: true (to train) or false (to evaluate)", type=str)
args = parser.parse_args()
if args.train=="true":
    train = True
elif args.train=="false":
    train = False
else:
    raise NameError("{} is not supported".format(args.train))

#Decomment for debugging
show_debug = False
cleaned_data_dir = r"/data/"
list_ = []
relevant_features = ["Acceleration", "Steering", "Brake", "Speed"]

for subdir, dirs, files in os.walk(cleaned_data_dir):
    for file in files:
        if file.startswith('clean__'):
            csv_path = os.path.join(subdir, file)
            df = pd.read_csv(csv_path,index_col=None, header=0, sep=',')
            df = df[:-1] #drop last row to bring in sync with facial data
            list_.append(df)

def concat_ordered_columns(frames):
    columns_ordered = []
    for frame in frames:
        columns_ordered.extend(x for x in frame.columns if x not in columns_ordered)
    final_df = pd.concat(frames)    
    return final_df[columns_ordered]       


df_full = concat_ordered_columns(list_)
show_debug = True
if show_debug is True:
    print(df_full.shape)

#Drop NaN values
df_full = df_full.dropna()

if show_debug is True:
    print(df_full.shape)


fullData = df_full[relevant_features]
y_labels = df_full.iloc[:,23:31]

#drop Disgust label and it's not relevant for driving and too similar to neutral
y_labels = y_labels.drop(columns=['Disgust'])

if show_debug is True:
    print(fullData.shape)
    print(y_labels.shape)


#To have categorical labels, take the max of each row in the dataframe as the ground truth
maxLabels = y_labels.idxmax(axis=1)

maxLabels[maxLabels == 'Fear'] = '0'
maxLabels[maxLabels == 'Contempt'] = '2'
maxLabels[maxLabels == 'Anger'] = '3'
maxLabels[maxLabels == 'Sad'] = '1'
maxLabels[maxLabels == 'Surprise'] = '4'
maxLabels[maxLabels == 'Joy'] = '4'
maxLabels[maxLabels == 'Neutral'] = '4'
maxLabels = maxLabels.fillna('4')

if show_debug is True:
    print(len(maxLabels[maxLabels== '0']))
    print(len(maxLabels[maxLabels== '1']))
    print(len(maxLabels[maxLabels== '2']))
    print(len(maxLabels[maxLabels== '3']))
    print(len(maxLabels[maxLabels== '4']))

#Encode labels to Integer for One-hot encoding
le = LabelEncoder()
y_labels['Max'] = le.fit_transform(maxLabels)

#Normalize training and test data
trainingValues = fullData.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
trainingValues_scaled = min_max_scaler.fit_transform(trainingValues)
fullData = pd.DataFrame(trainingValues_scaled, columns=fullData.columns)

#Reconcatenate the y_labels to training data after normalization
fullData = fullData.reset_index(drop=True)
y_labels = y_labels.reset_index(drop=True)

#Can change to full labels if necessary, set to only the encoded label values currently.
fullData = pd.concat([fullData, y_labels], axis=1)
if show_debug is True:
    print(fullData.shape)

# drop the positive and negative emotion values as we only focus on negative emotions
fullData = fullData[fullData["Max"] != 4]
y_labels = fullData.iloc[:,4:12]

fullData = fullData.reset_index(drop=True)
y_labels = y_labels.reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(fullData[relevant_features], y_labels, test_size=0.20, random_state=1, shuffle=False)

x_train_array = []
y_train_array = []
x_test_array = []
y_test_array = []
x_train_array.append(x_train)
y_train_array.append(y_train)
x_test_array.append(x_test)
y_test_array.append(y_test)

if show_debug is True:
    print(len(x_train))
    print(len(y_train))

def create_segments_and_labels(df, time_steps, step, labeldf):
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
    labels = []
    for j in range(0, len(df)):
        for i in range(1, len(df[j]) - time_steps, step):
            steering = df[j]['Steering'].values[i: i + time_steps]
            acceleration = df[j]['Acceleration'].values[i: i + time_steps]
            speed = df[j]['Speed'].values[i: i + time_steps]
            brake = df[j]['Brake'].values[i: i + time_steps]

            segments.append([steering, acceleration, speed, brake])

            maxLabel = labeldf[j]['Max'].values[i: i + time_steps]
            labels.append(maxLabel)

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)

        return reshaped_segments, labels


TIME_PERIODS = 30
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 1


x_train, y_train = create_segments_and_labels(x_train_array,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              y_train_array
                                              )

#Can uncomment if y is a 2D vector
y_train = y_train[:, 0]
if show_debug:
    print(np.shape(x_train))
    print(x_train.shape[0], 'training samples')
    print(np.shape(y_train))
    print(np.shape(x_test))


# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
#labels to train on
num_classes = 4

input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

if show_debug is True:
    print('x_train shape:', x_train.shape)
    print('input_shape:', input_shape)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")

# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)


# Add class weights because the model is extremely unbalanced
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))


config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if train is True:
    # 1D CNN neural network
    
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    
    model.add(Convolution1D(filters=16, kernel_size=(7), padding='same',
                            name='image_array', input_shape=(TIME_PERIODS, num_sensors)))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=16, kernel_size=(7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=(2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution1D(filters=32, kernel_size=(5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=32, kernel_size=(5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=(2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution1D(filters=64, kernel_size=(3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=64, kernel_size=(3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=(3), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution1D(filters=128, kernel_size=(3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=128, kernel_size=(3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=(3), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution1D(filters=128, kernel_size=(3), padding='same'))
    model.add(BatchNormalization(name = 'feature_extractor'))
    model.add(Convolution1D(
        filters=num_classes, kernel_size=(3), padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax'))

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='./models/behavioral_CNN/CNN_new_behavioral_negatives.h5',
            monitor='val_loss', save_best_only=True),
        #keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]
    
    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000001, amsgrad=False)
    BATCH_SIZE = 16
    EPOCHS = 50

    with sess.as_default():
        model.compile(loss='categorical_crossentropy',
                    optimizer=adam, metrics=['accuracy'])

        history = model.fit(x_train,
                          y_train,
                          class_weight = d_class_weights,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks_list,
                          validation_split=0.2,
                          verbose=True)

        with open('./trainHistoryDictBehav', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    # summarize history for accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.savefig('training_curve_behavioural_new.png')
    #plt.show()
else:
    #load model if desired
    model = load_model('./models/behavioral_CNN/CNN_new_behavioral_negatives.h5')



class_names = ['Fear', 'Sad', 'Contempt', 'Anger']
x_test, y_test = create_segments_and_labels(x_test_array,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                           y_test_array)


y_test = y_test[:, 0]

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

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
    plt.savefig('./confusion_matrix_behav.png')
    print("Confusion matrix was saved to confusion_matrix_behav.png'")
    #plt.show()

with sess.as_default():
    sess.run(tf.global_variables_initializer())
    score = model.evaluate(x_test, y_test, verbose=1)

    print("\nAccuracy on test data: %0.2f" % score[1])
    print("\nLoss on test data: %0.2f" % score[0])

    y_pred_test = model.predict(x_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    create_confusion_matrix(max_y_test, max_y_pred_test)


print("\n--- Classification report for test data ---\n")
print(classification_report(max_y_test, max_y_pred_test))


#Save classification report to file
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
df_class_report.to_csv('./class_report_behav.csv',  sep=',')





