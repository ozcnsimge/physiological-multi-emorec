#!/usr/bin/env python
"""
This code loads the model from the pretrained Behavioral, Physiological, and Facial Trained Networks
and returns the features associated with each. This is then fed into a CNN-LSTM to end up with the final prediction.
For evaluation pretrained fusion network can also be loaded.

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
parser.add_argument("train", help="Train or evaluate a pretrained model, options: true (to train) or false (to evaluate)", type=str)
args = parser.parse_args()
if args.train=="true":
    train = True
elif args.train=="false":
    train = False
else:
    raise NameError("{} is not supported".format(args.train))

phsio_model_name = "stresnet"
load_extracted_inputs = False
show_debug = False
cleaned_data_dir = r"/data"
list_ = []
relevant_features = ["Palm.EDA","Heart.Rate","Breathing.Rate","Perinasal.Perspiration","Acceleration", "Steering", "Brake", "Speed"]

for subdir, dirs, files in os.walk(cleaned_data_dir):
    for file in files:
        if file.startswith('clean__'):
            csv_path = os.path.join(subdir, file)
            df = pd.read_csv(csv_path,index_col=None, header=0, sep=',')
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

#Drop NaN values
df_full = df_full.dropna()

#All Data
fullData = df_full[relevant_features]
y_labels = df_full.iloc[:,23:31]

#drop Disgust label and it's not relevant for driving and too similar to neutral
y_labels = y_labels.drop(columns=['Disgust'])

#To have categorical labels, take the max of each row in the dataframe as the ground truth
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
y_original = y_labels.to_numpy()

#Can change to full labels if necessary, set to only the encoded label values currently.
fullData = pd.concat([fullData, y_labels], axis=1)

# Drop positive and neutral emotions as we only focus on negative emotions
fullData = fullData[fullData["Max"] != 4]
if show_debug is True:
    print(fullData.shape)
y_labels = fullData.iloc[:,8:16]

#Create Training and Test set
x_train, x_test, y_train, y_test = train_test_split(fullData[relevant_features], y_labels, test_size=0.2, random_state=1, shuffle=False)

physiological_features = ["Palm.EDA","Heart.Rate","Breathing.Rate","Perinasal.Perspiration"]
behavioral_features = ["Acceleration", "Steering", "Brake", "Speed"]

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
x_train_behavioral, x_test_behavioral = create_list(x_train[behavioral_features], x_test[behavioral_features])


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
            palm_r = np.asarray(palm, dtype= np.float32).reshape(-1, time_steps, 1)
            hr_r = np.asarray(hr, dtype= np.float32).reshape(-1, time_steps, 1)
            br_r = np.asarray(br, dtype= np.float32).reshape(-1, time_steps, 1)
            per_r = np.asarray(per, dtype= np.float32).reshape(-1, time_steps, 1)
            list_f = []
            list_f.append(palm_r)
            list_f.append(hr_r)
            list_f.append(br_r)
            list_f.append(per_r)
            labels = np.asarray(labels)

            return list_f, labels

        # Bring the segments into a better shape
        print(len(segments[0]))
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)

        return reshaped_segments, labels


#Variables for input to CNN
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
x_train_behavioral, y_train = create_segments_and_labels(x_train_behavioral,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              y_train_array,
                                              True
                                              )

x_test_physio, y_test = create_segments_and_labels(x_test_physio,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              y_test_array,
                                              False
                                              )
x_test_behavioral, y_test = create_segments_and_labels(x_test_behavioral,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              y_test_array,
                                              True
                                              )


#Can uncomment if y is a 2D vector
y_train = y_train[:, 0]
y_test = y_test[:, 0]


if show_debug:
    print(np.shape(x_train_behavioral))
    print(np.shape(x_train_physio[0]))
    print(x_train.shape[0], 'training samples')
    print(np.shape(y_train))

# Set input & output dimensions
num_time_periods, num_sensors = x_train_behavioral.shape[1], x_train_behavioral.shape[2]
#labels to train on
num_classes = 4

input_shape = (num_time_periods*num_sensors)

# Set input_shape / reshape for Keras
x_train_behavioral = x_train_behavioral.reshape(x_train_behavioral.shape[0], input_shape)
x_test_behavioral = x_test_behavioral.reshape(x_test_behavioral.shape[0], input_shape)

if show_debug is True:
    print('x_train shape:', x_train_behavioral.shape)
    print('input_shape:', input_shape)

# Convert type for Keras otherwise Keras cannot process the data
x_train_behavioral = x_train_behavioral.astype("float32")
x_train_physio = [x.astype("float32") for x in x_train_physio]
x_test_behavioral = x_test_behavioral.astype("float32")
x_test_physio = [x.astype("float32") for x in x_test_physio]


# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print('New y_train shape: ', y_train.shape)
print('New y_test shape: ', y_test.shape)

def preprocess_input(x, v2=True, imresize=True):
    
    if imresize:
        x = np.resize(x, (64,64))
        
    x = x.astype('float32')
    x = x / 255.0    
    
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def behavioral_model(behavioral_data):
    layer_name = 'feature_extractor'

    model_m = load_model('./models/behavioral_CNN/CNN_new_behavioral_negatives.h5')

    intermediate_layer_model = Model(inputs=model_m.input,
                                outputs=model_m.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(behavioral_data)
    return intermediate_output

def physiological_model(physiological_data):
    model = load_model('./models/physiological_CNN/stresnet.h5', custom_objects={'Spectrogram':kapre.time_frequency.Spectrogram})
    model.layers.pop()  # pop the last layer to get to the global avg pooling layer
                        
    model.outputs = [model.layers[-1].output]
    model.output_layers = [model.layers[-1]]
    model.layers[-1].outbound_nodes = []

    intermediate_output = model.predict(physiological_data)
    return intermediate_output

def facial_model(clean_path, concat=False):
    """
    Inputs all Videos from the cleaned directory and feeds them through the network
    Outputs a combined feature vector

    """
    emotion_model_path = r"./models/facial_CNN/facialnew_CNN_negatives_mini_XCEPTION_2018TRAINED4Sentiment.hdf5"
    extractor = load_model(emotion_model_path, compile=False)
    extractor.layers.pop()  # pop the last layer to get to the global avg pooling layer

    extractor.outputs = [extractor.layers[-1].output]
    extractor.output_layers = [extractor.layers[-1]]
    extractor.layers[-1]._outbound_nodes = []

    featuremap = []

    for subdir, dirs, files in os.walk(clean_path):
        dirs.sort()
        print("success, jumping to next directory:   " + str(subdir))
        for file in sorted(files):
            if file.endswith('.avi'):
                video_file_path = os.path.join(subdir, file)
                print("Opening File:  " + str(video_file_path))
                video_capture = cv2.VideoCapture(video_file_path)


                while video_capture.isOpened():

                    success, bgr_image = video_capture.read()

                    if not success:
                        break

                    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

                    gray_face = preprocess_input(gray_image)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)

                    feature = extractor.predict(gray_face) # This is what we want

                    features = feature[0]
                    featuremap.append(feature)


                video_capture.release()

    if concat:
        featuremap = np.concatenate(featuremap, axis=0)

    return featuremap



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

if load_extracted_inputs is False:
    with session.as_default():
        clean_path_facial = r"/data"
        behavioral_tensor = behavioral_model(x_train_behavioral)
        physiological_tensor = physiological_model(x_train_physio)
        behavioral_tensor_test = behavioral_model(x_test_behavioral)
        physiological_tensor_test = physiological_model(x_test_physio)
        facial_tensor = facial_model(clean_path_facial, True)
        np.save(r'./models/fusion_model/behavioral_tensor_test.npy', behavioral_tensor_test)
        np.save(r'./models/fusion_model/physiological_tensor_test.npy', physiological_tensor_test)
        np.save(r'./models/fusion_model/facial_tensor', facial_tensor)
        np.save(r'./models/fusion_model/behavioral_tensor', behavioral_tensor)
        np.save(r'./models/fusion_model/physiological_tensor', physiological_tensor)
else:
    facial_tensor = np.load(r'./models/fusion_model/facial_tensor.npy')
    behavioral_tensor = np.load(r'./models/fusion_model/behavioral_tensor.npy')
    physiological_tensor = np.load(r'./models/fusion_model/physiological_tensor_{}.npy'.format(phsio_model_name))
    behavioral_tensor_test = np.load(r'./models/fusion_model/behavioral_tensor_test.npy')
    physiological_tensor_test = np.load(r'./models/fusion_model/physiological_tensor_test_{}.npy'.format(phsio_model_name))

if show_debug is True:
    print(facial_tensor.shape)
    print(behavioral_tensor.shape)
    print(physiological_tensor.shape)
    print(behavioral_tensor_test.shape)
    print(physiological_tensor_test.shape)

behavioral_tensor = np.squeeze(behavioral_tensor)
physiological_tensor = np.squeeze(physiological_tensor)
behavioral_tensor_test = np.squeeze(behavioral_tensor_test)
physiological_tensor_test = np.squeeze(physiological_tensor_test)

new_facial_tensor = np.delete(facial_tensor, indsNaN, axis=0)

idx = np.where(y_original==4)
new_facial_tensor = np.delete(new_facial_tensor, idx, axis=0)
facial_tensor_train = new_facial_tensor[0:91824, :]
facial_tensor_test = new_facial_tensor[91884:-1, :]


def combined_model(behavioral_frames, physiological_frames, facial_frames):
    """Creates the behavioral-physiological-facial model.
    Args:
        behavioral_frames: A tensor that contains the behavioral input.
        physiological_frames: A tensor that contains the physiological input.
        facial_frames: A tensor that contains the facial input.
    Returns:
        The combined model.
    """
    behavioral_frames = behavioral_frames
    physiological_frames = physiological_frames
    return np.concatenate([behavioral_frames, physiological_frames, facial_frames], axis=1)


combined_tensor = combined_model(behavioral_tensor, physiological_tensor, facial_tensor_train)

combined_tensor_test = combined_model(behavioral_tensor_test, physiological_tensor_test, facial_tensor_test)
if show_debug is True:
    print(combined_tensor.shape)
    print(combined_tensor_test.shape)

#Normalize training and test data
min_max_scaler = preprocessing.MinMaxScaler()
combined_tensor = min_max_scaler.fit_transform(combined_tensor)
combined_tensor_test = min_max_scaler.fit_transform(combined_tensor_test)

# Add class weights because the model is extremely unbalanced
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

if train is True:
    model = Sequential()
    model.add(Reshape((-1, 128, 3), input_shape=(384,)))
    model.add(TimeDistributed(Convolution1D(filters=16, kernel_size=(7), padding='same',
                            input_shape=(128, 3))))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Convolution1D(filters=16, kernel_size=(7), padding='same')))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling1D(pool_size=(2), padding='same')))
    model.add(Dropout(.5))

    model.add(TimeDistributed(Convolution1D(filters=32, kernel_size=(5), padding='same')))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Convolution1D(filters=32, kernel_size=(5), padding='same')))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling1D(pool_size=(2), padding='same')))
    model.add(Dropout(.5))

    model.add(TimeDistributed(Convolution1D(filters=64, kernel_size=(3), padding='same')))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Convolution1D(filters=64, kernel_size=(3), padding='same')))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling1D(pool_size=(3), padding='same')))
    model.add(Dropout(.5))

    model.add(TimeDistributed(Convolution1D(filters=128, kernel_size=(3), padding='same')))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Convolution1D(filters=128, kernel_size=(3), padding='same')))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling1D(pool_size=(3), padding='same')))

    model.add(TimeDistributed(Convolution1D(
        filters=num_classes, kernel_size=(3), padding='same')))
    model.add(TimeDistributed(GlobalAveragePooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(.5))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(num_classes, activation='softmax'))
else:
    model = load_model(r'./models/fusion_model/fusion_negatives_facial_behav_{}.h5'.format(phsio_model_name))

relevant_modalities = ['behavioral', 'physiological', 'facial']
model.summary()
callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='./models/fusion_model/fusion_negatives_facial_behav_{}.h5'.format(phsio_model_name),
            monitor='val_acc', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=15)
    ]

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000001, amsgrad=False)


x_train = combined_tensor
y_train_total = y_train
print(x_train.shape)
print(y_train_total.shape)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

if train:
    history = model.fit(x_train,
                         y_train_total,
                         callbacks=callbacks_list,
                         batch_size=128,
                         epochs=50,
                         verbose=2,
                         validation_split=0.2,
                         class_weight = d_class_weights #Train with balanced classes to compensate for more neutral
                         )
    
    with open('trainHistory_fusion_negatives_facial_behav_{}'.format(phsio_model_name), 'wb+') as file_pi:
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
    plt.savefig('./training_curve_fusion_negatives_{}.png'.format(phsio_model_name))


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
    plt.savefig('./confusion_matrix_fusion_negatives_facial_behav_{}.png'.format(phsio_model_name))

score_train = model.evaluate(combined_tensor, y_train, verbose=1)
score = model.evaluate(combined_tensor_test, y_test, verbose=1)

print("\nAccuracy on train data: %0.2f" % score_train[1])
print("\nLoss on train data: %0.2f" % score_train[0])
print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

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
df_class_report.to_csv('./classification_report_fusion_negatives_facial_behav_{}.csv'.format(phsio_model_name),  sep=',')

