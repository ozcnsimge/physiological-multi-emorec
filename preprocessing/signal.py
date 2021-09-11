class Signal:
    def __init__(self, name, sampling, data):
        self.name = name
        self.sampling = sampling
        self.data = data


class NoSuchSignal(Exception):
    def __init__(self, signal_name):
        self.message = f"No such signal: {signal_name}"
