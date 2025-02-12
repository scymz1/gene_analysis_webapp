from torch.utils.data import Dataset, DataLoader, random_split
import re
import torch

import os
import numpy as np

class BioDataset(Dataset):
    def __init__(self, args):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        ebd_all = []
        lbl_all = []

        files_list_all = os.listdir(args.data_path)
        for f in files_list_all:
            print('f:', f)
            # 加载 embeddings
            ebd_file = np.load(os.path.join(args.data_path, f))
            ebd = ebd_file['array'] if 'array' in ebd_file else ebd_file  # 处理不同的 npz 格式

            # 加载对应的标签
            label_file = "_".join(f.split("_")[:-1]) + "_labels.npz"
            print('label_file:', label_file)
            lbl_file = np.load(os.path.join(args.label_path, label_file))
            lbl = lbl_file['array'] if 'array' in lbl_file else lbl_file
            lbl_onehot = np.eye(2)[lbl]

            # 转换为 tensor 并添加到列表
            ebd_all.extend([torch.tensor(e, dtype=torch.float32) for e in ebd])
            lbl_all.extend([torch.tensor(l, dtype=torch.float32) for l in lbl_onehot])

        self.embeds = ebd_all
        self.labels = lbl_all
        assert len(ebd_all) == len(lbl_all)

        self.length = len(self.embeds)
        print('number of samples:', self.length)

    def __getitem__(self, item):
        return self.embeds[item], self.labels[item]

    def __len__(self):
        return self.length



def get_dataloaders(args):
    ebddataset = BioDataset(args)

    train_size = int(args.train_rate * len(ebddataset))
    test_size = len(ebddataset) - train_size

    train_dataset, test_dataset = random_split(ebddataset, [train_size, test_size])

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_data_loader, test_data_loader






