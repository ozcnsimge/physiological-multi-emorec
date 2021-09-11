#!/usr/bin/env python
"""
Preprocess the dataset "A multimodal dataset for various forms of distracted driving"
Labe the data with only negative emotions, apply preprocessing steps to
pEDA, perEDA, HR and BR signals
"""
import itertools as it
import csv
import os

import numpy as np
import pandas as pd
import scipy.stats

from preprocessing.helpers import filter_signal
from preprocessing.preprocessor import Preprocessor
from preprocessing.signal import Signal, NoSuchSignal
from preprocessing.subject import Subject


class DistractedDriving(Preprocessor):
    # TODO: Find a more elegant way to set the IDs
    IDS = [20, 27, 29, 31, 41, 43, 44, 50, 66, 73, 86]
    SUBJECTS_IDS = list(it.chain(range(3,7), range(8, 11), range(12, 15), range(16, 19), range(22, 26), range(33, 37), range(38, 40), range(47, 48),
                range(54, 56), range(60, 63), range(75, 78), range(79, 82), range(83, 85), IDS))


    def __init__(self, logger, path):
        Preprocessor.__init__(self, logger, path, "DistractedDriving", [], None, subject_cls=DistractedDrivingSubject)

    def get_subjects_ids(self):
        return [f"{i:03}" for i in self.SUBJECTS_IDS]


def original_sampling(channel_name: str):
    if channel_name.startswith("Palm.EDA"):
        return 25

    if channel_name.startswith("Heart.Rate"):
        return 1

    if channel_name.startswith("Breathing.Rate"):
        return 1

    if channel_name.startswith("Perinasal.Perspiration"):
        return 7.5

    if channel_name == "label":
        return 25

    raise NoSuchSignal(channel_name)


def label_facs_data(emotion_data):
    df = emotion_data.drop(['Disgust'], axis=1)
    # Label the rows
    df['Emotion'] = df.idxmax(axis=1)
    for index, row in df.iterrows():
        if row['Emotion'] == 'Fear':
            df.loc[index, 'Emotion'] = 0
        if row['Emotion'] == 'Sad':
            df.loc[index, 'Emotion'] = 1
        if row['Emotion'] == 'Contempt':
            df.loc[index, 'Emotion'] = 2
        if row['Emotion'] == 'Anger':
            df.loc[index, 'Emotion'] = 3
        if row['Emotion'] == 'Neutral' or row['Emotion'] == 'Joy' or row['Emotion'] == 'Surprise':
            df.loc[index, 'Emotion'] = 4

    df = df.rename(columns={"Emotion": "label"})
    # labels = df['label'].to_numpy()
    # labels_arr = np.append(labels_arr, labels)
    return df['label']


class DistractedDrivingSubject(Subject):
    def __init__(self, logger, path, subject_id, channels_names, get_sampling_fn):
        Subject.__init__(self, logger, path, subject_id, channels_names, get_sampling_fn)
        self._logger = logger
        self._path = path
        self.id = subject_id

        data, labels = self._load_subject_data_from_file()
        self._data = self._restructure_data(data, labels)
        self._process_data()

    def _process_data(self):
        data = self._filter_all_signals(self._data)
        self._create_sliding_windows(data)

    def _load_subject_data_from_file(self):
        self._logger.info("Loading data for subject {}".format(self.id))
        #df_facs = self._load_facs_values_from_file()
        data, labels = self.load_subject_data_from_file(self._path, self.id)
        self._logger.info("Finqished loading data for subject {}".format(self.id))

        return data, labels

    @staticmethod
    def load_subject_data_from_file(path, id):
        subject_path = "{0}/clean_T{1}".format(path, id)
        list_ = pd.DataFrame()
        relevant_features = ["Palm.EDA", "Heart.Rate", "Breathing.Rate", "Perinasal.Perspiration"]
        emotions = ["Anger","Contempt","Disgust","Fear","Joy","Sad","Surprise","Neutral"]
        for subdir, dirs, files in os.walk(subject_path):
            for file in sorted(files):
                if file.startswith('clean__'):
                    print("reading {}".format(file))
                    csv_path = os.path.join(subject_path, file)
                    df = pd.read_csv(csv_path, index_col=None, header=0, sep=',')
                    list_ = list_.append(df, ignore_index=True, sort=False)

        full_data = list_[relevant_features]
        emotion_data = list_[emotions]
        label_data = label_facs_data(emotion_data)

        #merge facs values here
        result = pd.concat([full_data, label_data], axis=1)
        # drop the non negative emotions
        result = result[result["label"] != 4]
        # drop nan rows
        result = result.dropna()
        #seperate the labels from the data
        labels = result['label'].to_numpy()
        result['label'].to_csv("{0}/labels/T{1}-labels.csv".format(path, id), mode='w+', index=False)
        result = result.drop(['label'], axis=1)

        result.to_csv("{0}/T{1}-trimmed.csv".format(path, id), mode='w+', index=False)

        data = {}
        reader = csv.DictReader(open("{0}/T{1}-trimmed.csv".format(path, id)))
        for column, value in next(reader).items():
            data.setdefault(column, []).append(value)

        temp_data = np.genfromtxt("{0}/T{1}-trimmed.csv".format(path, id), delimiter=',')
        result = temp_data.T

        for i, key in enumerate(data):
            data[key] = result[i]

        return data, labels


    def _restructure_data(self, data, labels):
        self._logger.info("Restructuring data for subject {}".format(self.id))
        signals = self.restructure_data(data, labels)
        self._logger.info("Finished restructuring data for subject {}".format(self.id))

        return signals

    @staticmethod
    def restructure_data(data, labels):
        new_data = {'label': labels, "signal": data}
        return new_data

    def _filter_all_signals(self, data):
        self._logger.info("Filtering signals for subject {}".format(self.id))
        signals = data["signal"]
        for signal_name in signals:
            signals[signal_name] = filter_signal(signals[signal_name], original_sampling(signal_name))
        self._logger.info("Finished filtering signals for subject {}".format(self.id))
        return data

    # TODO: Vectorize this
    def _create_sliding_windows(self, data):
        self._logger.info("Creating sliding windows for subject {}".format(self.id))

        self.x = [Signal(signal_name, 1, []) for signal_name in data["signal"]]

        sub_window_size = 30
        start = 1

        for i in range(len(data["signal"]["Palm.EDA"]) - sub_window_size):
            label_id = scipy.stats.mstats.mode(data["label"][start + i:start + sub_window_size + i])[0][0]

            if label_id not in [0, 1, 2, 3]:
                continue

            channel_id = 0
            for signal in data["signal"]:
                self.x[channel_id].data.append(data["signal"][signal][start + i:start + sub_window_size + i])
                channel_id += 1
            
            self.y.append(label_id)

        self._logger.info("Finished creating sliding windows for subject {}".format(self.id))
