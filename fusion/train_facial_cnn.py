#!/usr/bin/env python
"""
This code trains a Convolutional Neural Network on the synced and preprocessed data.
Loads and feeds all the video data, and predicts emotional status from only facial data.

- It requires to have preprocessed data in 'clean_data'
"""
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
import keras

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import to_categorical

from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_fscore_support, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from cnn import mini_XCEPTION

parser = argparse.ArgumentParser()
parser.add_argument("train", help="Train or evaluate a pretrained model, options: true (to train) or false (to evaluate)", type=str)
args = parser.parse_args()
if args.train=="true":  
    train = True
elif args.train=="false":
    train = False
else:
    raise NameError("{} is not supported".format(args.train))

load = False

def label_extractor(Lablefile, analytics = False):
    # drops everything except the highest label and the current time as a numpy 2xN array
    # also drop "disgust" as the labeling seems very badly on this one
    # furhtermore turn "anger, contempt and fear" into "bad", "joy and suprise" into "happy" and "neutral into neutral
    collums_to_Keep = ['Anger', 'Contempt', "Fear", "Joy", "Sad", "Surprise", "Neutral"]

    # read file
    df = pd.read_csv(Lablefile, index_col=None, header=0)
    
    
    # kick out everything that is not needed
    df_labels = df[collums_to_Keep]
    
    if analytics:
        overall_len = len(df)
        label_len = len(df_labels)
        
        if overall_len - label_len != 0:
            print(Lablefile)
            print("overall_len: ", overall_len)
            print("label_len: ", label_len)
        
    #To have categorical labels, take the max of each row in the dataframe as the ground truth
    maxLabels = df_labels.idxmax(axis=1)
    maxLabels = maxLabels.fillna('Neutral')
    
    maxLabels = maxLabels[:-1]
        
    #Convert categorical labels to Positive, Negative or Neutral
    maxLabels[maxLabels == 'Fear'] = 'Fear'
    maxLabels[maxLabels == 'Contempt'] = 'Contempt'
    maxLabels[maxLabels == 'Anger'] = 'Anger'
    maxLabels[maxLabels == 'Sad'] = 'Sad'
    maxLabels[maxLabels == 'Surprise'] = 'Surprise'
    maxLabels[maxLabels == 'Joy'] = 'Joy'
    maxLabels[maxLabels == 'Neutral'] = 'Neutral'

    # generate a dataframe with the list of emotions on each timeframe
    return maxLabels



def process_emotion(emotion):
    """
    Takes in a vector of emotions and outputs a list of emotions as one-hot vectors.
    :param emotion: vector of strings (happy, bad, neutral
    :return: list of one-hot vectors (array of 3)
    """
    data = []
    
    num_emotions = 5
    for i in range(len(emotion)):
        if emotion[i] == 'Fear' :
            data.append(0)
        if emotion[i] == 'Contempt':
            data.append(2)
        if emotion[i] == 'Anger':
            data.append(3)
        if emotion[i] == 'Sad':
            data.append(1)
        if emotion[i] == 'Surprise':
            data.append(4)
        if emotion[i] == 'Joy':
            data.append(4)
        if emotion[i] == 'Neutral':
            data.append(4)
        
    return data

def csv_emotion_2_one_hot (csv_path):
    """
    Turn a csv file into a list of one_hots
    :param csv_path: 
    :return: one_hot_emotion list  
    """
    # Load the csv File with only the lables
    df = label_extractor(csv_path)
    
    # Turn emotions to one_hot list
    one_hot_emotions = process_emotion(df)
    
    return one_hot_emotions

def load_all_emotions(path):
    """
    # Path to cleaned up lables and Videos
    path should be = r"*\clean-video-and-lables"    
    """
    
    emotion_train = []

    for subdir, dirs, files in os.walk(path):
        dirs.sort()
        for file in sorted(files):

            if file.startswith('clean__'):
                print("Current Direcoty,  " + str(subdir), end="\r")
                csv_path = os.path.join(subdir, file)
                one_hot = csv_emotion_2_one_hot(csv_path)
                emotion_train.append(one_hot)
    return emotion_train


def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)



# Load haarcascade facial detection    
detection_model_path = r"models/facial_CNN/haarcascade_frontalface_default.xml"
face_detection = load_detection_model(detection_model_path)

def face_detect(frame):
    # convert to greyscale      
    # face cutoff as (10px,10px)
    face_size = (64, 64)
    face_cuttoff = (10,10)

    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Facial detection algorithm with haarcascade
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, face_cuttoff)
        frame = gray_image[y1:y2, x1:x2]       
        
    return frame

face_cascade = cv2.CascadeClassifier('models/facial_CNN/haarcascade_frontalface_default.xml')


def split_frames(video_file_path, preprocess=True, no_face = True, detect_all = False, resize = (64, 64)):
    """ Splits video sequences into frames
    """
    cap = cv2.VideoCapture(video_file_path)
    frames = []
    counter = 0
    x1_old, x2_old, y1_old, y2_old = 1,1,1,1
    x1, x2, y1, y2 = 0,0,0,0
    while(cap.isOpened()):
        ret, frame = cap.read()   
        counter +=1
        if ret:
            if no_face:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.resize(gray_image, resize)

                frames.append(gray_image)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ('q'):
                    break
            else:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Facial detection algorithm with haarcascade
                faces = detect_faces(face_detection, gray_image)
                
                if detect_all:
                    for face_coordinates in faces:
                        x1, x2, y1, y2 = apply_offsets(face_coordinates, (10,10))  
                        
                        # Just make sure nothing goes wrong
                        # adjust values to your liking. sometimes the face detection goes wrong and so we revert to the face we used before
                        if y1 >= 100 or y1 <= 0:
        
                            x1, x2, y1, y2 = x1_old, x2_old, y1_old, y2_old
        
                        # zoom into the face
                        gray_image = gray_image[y1:y2, x1:x2]       
        
                        x1_old, x2_old, y1_old, y2_old = x1, x2, y1, y2
                else:
                    if counter == 1:
                        for face_coordinates in faces:
                            x1, x2, y1, y2 = apply_offsets(face_coordinates, (20,20))  

                    gray_image = gray_image[y1:y2, x1:x2]       

                    # resize the face to face_size
                    try:
                        gray_face = cv2.resize(gray_image, resize)
                    except:
                        continue

                    frames.append(gray_face)

                 # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    

    if preprocess == True:
        frames = preprocess_input(frames, expand_dims=True)


    return frames

def preprocess_input(x,expand_dims = False, v2=True):
    """
    
    :param x: array of  frames
    :param expand_dims: go from (x, 300,300) to (x, 300,300, 1) 
    :param v2: used to rescale the values between -1 and 1
    :return: processed frames
    """
    
    x = np.asarray(x) / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    if expand_dims:
        x = np.expand_dims(x, axis=3)
    return x


def load_training_data(path):
    """
    Loops over all preporcessed data and outputs all frames and emotions
    Loads and preprocesses training data
    """

    All_Frames = []
    
    for subdir, dirs, files in os.walk(path):
        dirs.sort()
        if subdir[-1].isdigit(): 
            print("Current Directory:  " + str(subdir))
            
        for file in sorted(files):
            if file.endswith('.avi'):
                video_file_path = os.path.join(subdir, file)

                frames = split_frames(video_file_path)
                
                All_Frames.append(frames)

    print ("finished loading all Videofiles. Now loading all lables")
    All_Emotions = load_all_emotions(path)
    print("done")
    return All_Frames, All_Emotions


def check_same_len(c_lables,c_frames):
    fail = False
    for i in range(len(c_lables)):
        a = len(c_frames[i])
        b = len(c_lables[i])
    
        # calculate the distance between them
        x = a-b
        
        if x !=0:
            print(i,x)
            fail = True
    
    if not fail:
        print("merging labels and frames works!")
        
  
    return c_frames

clean_path = r"/data"

if load is False:
    combined_frames_no_face, combined_lables = load_training_data(clean_path)

    cleaned_combined_frames = check_same_len(combined_lables,combined_frames_no_face)


num_emotions = 4

if load is False:
    # drop all the positive emotions
    print("processing the data.. this could take a while..")
    X = np.concatenate(combined_frames_no_face, axis = 0)
    Y = np.concatenate(combined_lables, axis = 0)
    idx = np.where(Y==4)
    X = np.delete(X, idx, axis=0)
    Y = np.delete(Y, idx, axis=0)

    Y = to_categorical(Y, num_emotions) 
    
    print("splitting the data to test and train")
    # Splitting and shuffling the training Data
    train_faces, test_faces, train_emotions, test_emotions = train_test_split(X,Y, test_size= 0.2, random_state=1, shuffle=False)      
    print('splitting successful')

    np.save('models/facial_CNN/combined_frames', X)
    np.save('models/facial_CNN/combined_labels', Y)
else:
    print("loding the frames and labels..")
    X = np.load(r'models/facial_CNN/combined_frames.npy')
    Y = np.load(r'models/facial_CNN/combined_labels.npy')
    train_faces, test_faces, train_emotions, test_emotions = train_test_split(X,Y, test_size= 0.2, random_state=1, shuffle=False)
    print(len(train_faces))
    print(len(train_emotions))


config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# parameters
batch_size = 16
num_epochs = 50
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 4
patience = 50

# Add class weights because the model is extremely unbalanced
y_integers = np.argmax(Y, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)



# model parameters/compilation
if train:
    model = mini_XCEPTION(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
    model.summary()


    # callbacks
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    trained_models_path = r'./models/facial_CNN/facialnew_CNN_negatives_mini_XCEPTION_2018TRAINED4Sentiment.hdf5'
    model_checkpoint = ModelCheckpoint(trained_models_path, 'val_loss', verbose=1,
                                                    save_best_only=True)

    callbacks_list = [model_checkpoint, early_stop, reduce_lr]


    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        history = model.fit(train_faces,
                            train_emotions,
                            class_weight=d_class_weights,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            callbacks=callbacks_list,
                            validation_split=0.2,
                            verbose=2)

model = load_model(r'./models/facial_CNN/facialnew_CNN_negatives_mini_XCEPTION_2018TRAINED4Sentiment.hdf5')

print(test_faces.shape)
print(test_emotions.shape)

class_names = ['Fear', 'Sad', 'Contempt', 'Anger']

def create_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions, normalize='all')
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
    plt.savefig('./confusion_matrix_facial.png')

with sess.as_default():
    sess.run(tf.global_variables_initializer())
    score = model.evaluate(test_faces, test_emotions, verbose=1)

    print("\nAccuracy on test data: %0.2f" % score[1])
    print("\nLoss on test data: %0.2f" % score[0])

    print("\n--- Confusion matrix for test data ---\n")

    y_pred_test = model.predict(test_faces)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(test_emotions, axis=1)

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
df_class_report.to_csv('./confusion_matrix_facial.csv',  sep=',')

