import os
import math
import pickle
from time import time

import numpy as np
import torch
import torch.nn as nn

from base.BaseRecommender import BaseRecommender
from utils import Tool
from dataloader.DataBatcher import DataBatcher

class EASE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(EASE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.reg = model_conf['reg']

        self.device = device

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        start = time()
        log_dir = logger.log_dir

        # dataset iterator
        train_matrix = dataset.train_matrix

        # P = (X^T * X + λI)^−1
        G = train_matrix.transpose().dot(train_matrix).toarray()
        diag = np.diag_indices(self.num_items)
        G[diag] += self.reg
        P = torch.Tensor(G).inverse()

        # B = P * (X^T * X − diagMat(γ))
        self.enc_w = -P / torch.diag(P)
        self.enc_w[diag] = 0

        # Save
        with open(os.path.join(log_dir, 'best_model.p'), 'wb') as f:
            pickle.dump(self.enc_w, f, protocol=4)

        # Evaluate
        testl_score = evaluator.evaluate(self, mean=True)
        testl_score_str = ['%s=%.4f' % (k, testl_score[k]) for k in testl_score]
        logger.info(', '.join(testl_score_str))

        total_train_time = time() - start

        return testl_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_eval_pos = eval_pos_matrix[user_ids]
        eval_output = torch.Tensor(batch_eval_pos.toarray()) @ self.enc_w

        if eval_items is not None:
            eval_output[np.logical_not(eval_items)]=float('-inf')
        else:
            eval_output[batch_eval_pos.nonzero()] = float('-inf')

        return eval_output.numpy()

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            self.enc_w = pickle.load(f)