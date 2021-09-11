import sys

from experiment.hyperparametertuning import HyperparameterTuning
from experiment.distracteddrivingexperiment import DistractedDrivingExperimentNFold
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import set_available_gpus


def get_dataset(name):
    if name.startswith("distracteddriving_fold_"):
        return DistractedDrivingExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    raise Exception(f"No such dataset/experiment as {name}")


if __name__ == '__main__':
    try:
        _, gpu_id, dataset_name, clas, max_eval = sys.argv

        set_available_gpus(gpu_id)
        dataset = get_dataset(dataset_name)

        HyperparameterTuning(dataset, [gpu_id]).tune_one(clas, int(max_eval))

    except Exception as e:
        GLOBAL_LOGGER.exception(e)
        raise e
