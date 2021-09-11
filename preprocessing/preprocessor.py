#!/usr/bin/env python
"""
A base class for Preprocessing.
Could be utilized for different datasets
"""
import gzip
import pickle
from abc import ABC, abstractmethod

from preprocessing.dataset import Dataset
from preprocessing.subject import Subject


class Preprocessor(ABC):
    def __init__(self, logger, path, name, channels_names, get_sampling_fn, subject_cls=Subject):
        self._logger = logger
        self._path = path
        self._name = name
        self.subjects = [subject_cls(self._logger, self._path, id_, channels_names, get_sampling_fn) for id_ in
                         self.get_subjects_ids()]

    def get_dataset(self):
        return Dataset(self._name, self, self._logger)

    @abstractmethod
    def get_subjects_ids(self):
        raise NotImplementedError()
