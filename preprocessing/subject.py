from preprocessing.signal import Signal


class Subject:
    def __init__(self, logger, path, id_, channels_names, get_sampling_fn):
        self._logger = logger
        self._path = path
        self.id = id_

        self.x = [Signal(signal_name, get_sampling_fn(signal_name), []) for signal_name in channels_names]
        self.y = []
