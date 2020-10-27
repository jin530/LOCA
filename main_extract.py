# Import packages
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn

import utils.Constant as CONSTANT
from dataloader import UIRTDatset
from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed
from dataloader.DataBatcher import DataBatcher

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='saves/MultVAE/2_20200812-0108', metavar='P')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read configs
config = Config(main_conf_path=args.path, model_conf_path=args.path)
model_name = config.get_param('Experiment', 'model_name')

# dataset
dataset_name = config.get_param('Dataset', 'dataset')
dataset_type = CONSTANT.DATASET_TO_TYPE[dataset_name]
dataset = UIRTDatset(**config['Dataset'])


import model
MODEL_CLASS = getattr(model, model_name)

# build model
model = MODEL_CLASS(dataset, config['Model'], device)
model.restore(args.path)

model.eval()
test_eval_pos, test_eval_target, _ = dataset.test_data()

if model_name == 'MultVAE':
    user_embedding = model.user_embedding(test_eval_pos)

    emb_dir = os.path.join(dataset.data_dir, dataset.data_name, 'embedding')
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    emb_file = os.path.join(emb_dir, model_name + '_user.p')
    with open(emb_file, 'wb') as f:
        pickle.dump(user_embedding, f, protocol=4)
    config.save(emb_dir)
    print("embedding extracted!")

num_users = len(test_eval_target)
num_items = test_eval_pos.shape[1]
eval_users = np.arange(num_users)
user_iterator = DataBatcher(eval_users, batch_size=1024)
output = np.zeros((num_users, num_items))
for batch_user_ids in user_iterator:
    batch_eval_pos = test_eval_pos[batch_user_ids]
    batch_pred = model.predict(batch_user_ids, test_eval_pos)
    output[batch_user_ids] += batch_pred

# output = model.predict(dataset).detach().cpu().numpy()

output_dir = os.path.join(dataset.data_dir, dataset.data_name, 'output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_file = os.path.join(output_dir, model_name + '_output.p')
with open(output_file, 'wb') as f:
    pickle.dump(output, f, protocol=4)
config.save(output_dir)
print("output extracted!")
