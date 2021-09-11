from preprocessing.distracteddriving import DistractedDriving
from experiment.experiment import Experiment, prepare_experimental_setups_n_iterations, \
    n_fold_split

SIGNALS_LEN = 4


class DistractedDrivingExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split([f"{i:03}" for i in DistractedDriving.SUBJECTS_IDS])

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "DistractedDriving", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)
