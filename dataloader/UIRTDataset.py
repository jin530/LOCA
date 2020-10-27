import os
import pickle

import numpy as np
import torch
import scipy.sparse as sp

from base.Dataset import BaseDataset
from dataloader.DataLoader import load_data_and_info


class UIRTDatset(BaseDataset):
    def __init__(self, data_dir, dataset, min_user_per_item=1, min_item_per_user=1, leave_k=5, popularity_order=True):
        super(UIRTDatset, self).__init__(data_dir, dataset, min_user_per_item, min_item_per_user, leave_k, popularity_order)

        self.train_matrix, self.test_dict, self.user_id_dict, self.user_to_num_items, self.item_id_dict, self.item_to_num_users \
            = load_data_and_info(self.data_file, self.info_file)

        print("data loaded!")

        self.num_users = len(self.user_id_dict)
        self.num_items = len(self.item_id_dict)
        self.eval_neg_candidates = None

    def test_data(self):
        eval_pos = self.train_matrix
        eval_target = self.test_dict
        return eval_pos, eval_target, self.eval_neg_candidates

    def __str__(self):
        ret_str = '\n'
        ret_str += 'Dataset: %s\n' % self.data_name
        ret_str += '# of users: %d\n' % self.num_users
        ret_str += '# of items: %d\n' % self.num_items
        return ret_str
