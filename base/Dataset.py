import os
from dataloader.Preprocess import preprocess
import utils.Constant as CONSTANT


class BaseDataset:
    def __init__(self, data_dir, dataset, min_user_per_item=1, min_item_per_user=1, leave_k=5, popularity_order=True):
        """
        Dataset base class

        :param str data_dir: base directory of data
        :param str dataset: Name of dataset e.g. ml-100k
        """

        self.data_dir = data_dir
        self.data_name = dataset

        self.min_user_per_item = min_user_per_item
        self.min_item_per_user = min_item_per_user

        self.leave_k = leave_k
        self.popularity_order = popularity_order

        prefix = os.path.join(self.data_dir, self.data_name, self.data_name)
        self.raw_file = prefix + '.rating'
        self.file_prefix = self.generate_file_prefix()
        self.data_file = self.file_prefix + '.data'
        self.info_file = self.file_prefix + '.info'
        self.separator = CONSTANT.DATASET_TO_SEPRATOR[dataset]

        if not self.check_dataset_exists():
            print('preprocess raw data...')
            preprocess(self.raw_file, self.file_prefix, self.leave_k, self.min_item_per_user, self.min_user_per_item, self.separator, self.popularity_order)

    def check_dataset_exists(self):
        return os.path.exists(self.data_file) and os.path.exists(self.info_file)

    def generate_file_prefix(self):
        split = 'fixed'
        base_directory = os.path.join(self.data_dir, self.data_name, split)
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        sub_directory = []
        sub_directory.append('_mincnt_%d_%d' % (self.min_user_per_item, self.min_item_per_user))
        sub_directory.append('_k_%d' % self.leave_k)

        sub_directory = os.path.join(base_directory, '_'.join(sub_directory))
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)

        return os.path.join(sub_directory, 'data')

    def __str__(self):
        return 'BaseDataset'
