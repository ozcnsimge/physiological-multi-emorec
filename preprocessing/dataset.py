import os
import pickle
import numpy as np

class Dataset:
    def __init__(self, name, dset, logger):
        self.name = name
        self.dset = dset
        self._logger = logger

    def save(self, path):
        path = "{}/{}".format(path, self.name)
        os.makedirs(path, exist_ok=True)

        for subject in self.dset.subjects:
            self._logger.info("Saving data for subject {}".format(subject.id))

            for channel_id in range(len(subject.x)):
                filename = "{}/x_{}_{}.pkl".format(path, subject.id, channel_id)
                with open(filename, "wb") as f:
                    pickle.dump(subject.x[channel_id], f)

            filename = "{}/y_{}.pkl".format(path, subject.id)
            with open(filename, "wb") as f:
                pickle.dump(subject.y, f)

    def load(self, path: str, subject_ids: tuple, channels_ids: tuple):
        self._logger.info("Loading data for subjects {} and channels {}".format(subject_ids, channels_ids))

        path = "{}/{}/".format(path, self.name)
        x = [[] for i in range(max(channels_ids) + 1)]
        sampling = [-1 for i in range(max(channels_ids) + 1)]
        y = []

        for subject_id in subject_ids:
            for channel_id in channels_ids:
                channel = self._unpickle(path, "x_{}_{}.pkl".format(subject_id, channel_id))
                x[channel_id] += channel.data
                sampling[channel_id] = channel.sampling
            y += self._unpickle(path, "y_{}.pkl".format(subject_id))

        sampling = [i for i in sampling if i != -1]
        x = [i for i in x if i]

        self._logger.info("Finished loading data for subjects {} and channels {}".format(subject_ids, channels_ids))
        return x, y, sampling

    @staticmethod
    def _unpickle(path: str, filename: str):
        path += filename
        with open(path.format(path), "rb") as f:
            result = pickle.load(f)
        return result
