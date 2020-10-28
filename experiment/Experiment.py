import copy
from time import time

def train_model(model, dataset, evaluator, early_stop, logger, config):
    # train model with the given experimental setting
    logger.info('train start ... !')
    early_stop.initialize()
    test_score, train_time = model.train_model(dataset, evaluator, early_stop, logger, config)

    return test_score, train_time