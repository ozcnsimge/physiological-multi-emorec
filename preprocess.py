import configparser

from preprocessing.distracteddriving import DistractedDriving
from utils.loggerwrapper import GLOBAL_LOGGER

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    dataset = DistractedDriving(GLOBAL_LOGGER, config['Paths']['dd_dir']).get_dataset()
    dataset.save(config['Paths']['out_dir'])
