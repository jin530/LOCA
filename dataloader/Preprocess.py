import os
import math
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm
import pickle
from datetime import datetime

import numpy as np
import scipy.sparse as sp


def read_raw_UIRT(datapath, separator, order_by_popularity=True):
    """
    read raw data (ex. ml-100k.rating)

    return U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, num_users, num_items, num_ratings

    """

    print("Loading the dataset from \"%s\"" % datapath)
    with open(datapath, "r") as f:
        lines = f.readlines()

    num_users, num_items = 0, 0
    user_to_num_items, item_to_num_users = {}, {}
    user_id_dict, item_id_dict = {}, {}

    for line in lines:
        user_id, item_id, _, _ = line.strip().split(separator)
        try:
            user_id = int(user_id)
            item_id = int(item_id)
        except:
            continue

        if user_id not in user_id_dict:
            user_id_dict[user_id] = num_users
            new_user_id = user_id_dict[user_id]

            user_to_num_items[new_user_id] = 1
            num_users += 1
        else:
            new_user_id = user_id_dict[user_id]
            user_to_num_items[new_user_id] += 1

        # Update the number of ratings per item
        if item_id not in item_id_dict:
            item_id_dict[item_id] = num_items

            new_item_id = item_id_dict[item_id]
            item_to_num_users[new_item_id] = 1
            num_items += 1
        else:
            new_item_id = item_id_dict[item_id]
            item_to_num_users[new_item_id] += 1

    if order_by_popularity:
        user_id_dict, user_to_num_items = order_id_by_popularity(user_id_dict, user_to_num_items)
        item_id_dict, item_to_num_users = order_id_by_popularity(item_id_dict, item_to_num_users)

    # BUILD U2IRTs
    U2IRT = {u: [] for u in user_to_num_items}
    for line in lines:
        user_id, item_id, rating, time = line.strip().split(separator)
        try:
            user_id = int(user_id)
            item_id = int(item_id)
            rating = float(rating)
            time = int(time)
        except:
            continue
        U2IRT[user_id_dict[user_id]].append((item_id_dict[item_id], rating, time))
    return U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users


def order_id_by_popularity(object_id_dict, object_to_num):
    old_to_pop_dict = {}
    new_to_pop_dict = {}
    new_object_to_num = {}
    object_id_dict_sorted = sorted(object_to_num.items(), key=lambda x: x[-1], reverse=True)
    for pop, new_pop_tuple in enumerate(object_id_dict_sorted):
        new = new_pop_tuple[0]
        new_to_pop_dict[new] = pop
        new_object_to_num[pop] = object_to_num[new]
    for old, new in object_id_dict.items():
        old_to_pop_dict[old] = new_to_pop_dict[new]

    return old_to_pop_dict, new_object_to_num


def filter_min_item_cnt(U2IRT, min_item_cnt, user_id_dict):
    modifier = 0
    user_id_dict_sorted = sorted(user_id_dict.items(), key=lambda x: x[-1])
    for old_user_id, _ in user_id_dict_sorted:
        new_user_id = user_id_dict[old_user_id]
        IRTs = U2IRT[new_user_id]
        num_items = len(IRTs)

        if num_items < min_item_cnt:
            U2IRT.pop(new_user_id)
            user_id_dict.pop(old_user_id)
            modifier += 1
        elif modifier > 0:
            U2IRT[new_user_id - modifier] = IRTs
            user_id_dict[old_user_id] = new_user_id - modifier
    return U2IRT, user_id_dict


def preprocess(raw_file, file_prefix, leave_k, min_item_per_user=0, min_user_per_item=0, separator=',', order_by_popularity=True):
    """
    read raw data and preprocess

    """
    # read raw data
    U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users = read_raw_UIRT(raw_file, separator, order_by_popularity)

    # filter out (min item cnt)
    if min_item_per_user > 0:
        U2IRT, user_id_dict = filter_min_item_cnt(U2IRT, min_item_per_user, user_id_dict)

    # preprocess
    preprocess_lko(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, file_prefix, leave_k)


def preprocess_lko(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, file_prefix, leave_k):
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = 0

    train_data = np.zeros((num_users, num_items))
    test_dict = {u: [] for u in range(num_users)}

    for user in tqdm(U2IRT):
        # Sort the UIRTs by the ascending order of the timestamp.
        IRTs = sorted(U2IRT[user], key=lambda x: x[-1])
        num_ratings += len(IRTs)

        # test data
        for i in range(leave_k):
            IRT = IRTs.pop()
            test_dict[user].append(IRT[0])

        # train data
        for IRT in IRTs:
            train_data[user, IRT[0]] = 1

    train_sp_matrix = csr_matrix(train_data, shape=(num_users, num_items))

    # save
    data_to_save = {
        'train': train_sp_matrix,
        'test': test_dict,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    ratings_per_user = [len(U2IRT[u]) for u in U2IRT]

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f" % (
        min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)

    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess Leave-%d-out finished.' % (leave_k))
