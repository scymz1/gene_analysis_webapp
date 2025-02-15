
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)
from scipy.sparse import csr_matrix
import os
import copy
import scanpy
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import torch
from torch.utils.data import Dataset
import anndata
from open_biomed.feature.cell_featurizer import SUPPORTED_CELL_FEATURIZER

class CTCDataset(Dataset, ABC):
    def __init__(self, path, config, seed):
        super(CTCDataset, self).__init__()
        self.config = config
        self.path = path
        self.seed = seed
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        feat_config = self.config["cell"]["featurizer"]["structure"]
        featurizer = SUPPORTED_CELL_FEATURIZER[feat_config["name"]](feat_config)
        self.cells = [featurizer(cell) for cell in self.cells]

    # def index_select(self, indexes):
    #     new_dataset = copy.copy(self)
    #     new_dataset.cells = [self.cells[i] for i in indexes]
    #     new_dataset.labels = [self.labels[i] for i in indexes]
    #     return new_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.cells[index][:200], self.labels[index]

class Zheng68k(CTCDataset):
    def __init__(self, path, config, seed):
        super(Zheng68k, self).__init__(path, config, seed)
        self._train_test_split(seed)

    def _load_data(self):
        print('loading: '+self.path)
        csv_file_path = self.path
        # with open(csv_file_path, 'r') as f:
        #     first_line = f.readline()
        # total_columns = len(first_line.split('\t'))
        # # 2. 根据总列数设置需要读取的列范围
        # if total_columns > 19300:
        #     usecols = range(2, 19203)  # 读取第2列到第19002列（从0开始）
        # else:
        #     usecols = range(2, 19203)  # 读取第2列到最后一列
        df = pd.read_csv(csv_file_path, sep='\t', index_col=0,
                         usecols=range(1, 300))  # limitation of vocab 19379
        lbl = list(pd.read_csv(csv_file_path, sep='\t', usecols=['Condition'])['Condition'])
        label = [1 if element == 'sensitive' else 0 for element in lbl]

        adata = anndata.AnnData(df)
        adata.X = csr_matrix(adata.X.astype(np.float32))
        self.cells = adata.X
        self.labels = np.eye(2)[label]
        self.num_classes = 2


    def _train_test_split(self, seed):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(self.cells, self.labels)
        for train_index, val_index in split:
            self.train_index = train_index
            self.val_index = val_index


SUPPORTED_CTC_DATASETS = {
    "zheng68k": Zheng68k,
}