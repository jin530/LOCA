import math
import pickle

import numpy as np
import scipy.sparse as sp

def load_data_and_info(data_file, info_file):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    with open(info_file, 'rb') as f:
        info_dict = pickle.load(f)
        
    user_id_dict = info_dict['user_id_dict']
    user_to_num_items = info_dict['user_to_num_items']
    item_id_dict = info_dict['item_id_dict']
    item_to_num_users = info_dict['item_to_num_users']

    num_users = data_dict['num_users']
    num_items = data_dict['num_items']

    train_sp_matrix = data_dict['train']
    test_sp_matrix = data_dict['test']

    return train_sp_matrix, test_sp_matrix, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users
