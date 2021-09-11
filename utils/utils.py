import os

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


class NoSuchClassifier(Exception):
    def __init__(self, classifier_name):
        self.message = "No such classifier: {}".format(classifier_name)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, sampling_rates=None,
                      ndft_arr=None):
    if classifier_name == 'stresnetM':
        from classifier import spectroTemporalResNet_multimodal
        return spectroTemporalResNet_multimodal.ClassifierStresnet(output_directory, input_shape, sampling_rates,
                                                                   ndft_arr,
                                                                   nb_classes, verbose=verbose)

    raise NoSuchClassifier(classifier_name)


def prepare_data(x_train, y_train, y_val, y_test):
    y_train, y_val, y_test = transform_labels(y_train, y_test, y_val=y_val)
    y_true = y_val.astype(np.int64)
    concatenated_ys = np.concatenate((y_train, y_val, y_test), axis=0)
    nb_classes = len(np.unique(concatenated_ys))
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(concatenated_ys.reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    
    if type(x_train) == list:
        input_shapes = [x.shape[1:] for x in x_train]
    else:
        input_shapes = x_train.shape[1:]
    return input_shapes, nb_classes, y_val, y_train, y_test, y_true


def set_available_gpus(gpu_ids):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def get_new_session():
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def transform_labels(y_train, y_test, y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None:
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train, y_val, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val, new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

#TODO: Inspect and modify
def save_logs(output_directory, hist, y_pred, y_pred_probabilities, y_true, duration, lr=True, y_true_val=None,
              y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, y_pred_probabilities, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    accuracy_name = 'accuracy' if 'accuracy' in row_best_model else 'acc'
    df_best_model['best_model_train_acc'] = row_best_model[accuracy_name]
    df_best_model['best_model_val_acc'] = row_best_model['val_' + accuracy_name]
    # if lr == True:
    #     df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
    plot_predictions(y_pred, y_true, output_directory + 'predictions.png')
    save_predictions(y_true, y_pred, y_pred_probabilities, f"{output_directory}predictions.txt")

    return df_metrics


def plot_predictions(y_pred, y_true, filename):
    fig, ax = plt.subplots()
    t = list(range(len(y_pred)))
    ax.plot(t, y_true, "b-", t, y_pred, "r.")
    fig.savefig(filename)
    plt.close(fig)


def save_predictions(y_true, y_pred, y_pred_probabilities, filename):
    with open(filename, "w+") as file:
        for line in [y_true, y_pred, y_pred_probabilities]:
            for elem in line:
                file.write(f"{elem} ")
            file.write("\n")


def plot_epochs_metric(hist, file_name, metric='loss'):
    fig, ax = plt.subplots()
    ax.plot(hist.history[metric])
    ax.plot(hist.history['val_' + metric])
    ax.set_title('model ' + metric)
    ax.set_ylabel(metric, fontsize='large')
    ax.set_xlabel('epoch', fontsize='large')
    ax.legend(['train', 'val'], loc='upper left')
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)


def calculate_metrics(y_true, y_pred, y_pred_probabilities, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration', 'f1', 'auc'])
    res['duration'] = duration

    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['f1'] = f1_score(y_true, y_pred, average='macro')

    try:
        if y_pred_probabilities.shape[1] == 2:
            res['auc'] = roc_auc_score(y_true, y_pred_probabilities[:, 0], multi_class="ovo")
        else:
            res['auc'] = roc_auc_score(y_true, y_pred_probabilities, multi_class="ovo")
    except:
        res['auc'] = None

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)
    return res
