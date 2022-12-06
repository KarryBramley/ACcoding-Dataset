import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

from itertools import cycle

from ..utils.io import loadpkl
from .base_loader import BaseLoader

class OjLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        if config.use_mini:
            prefix = 'mini_'
        else:
            prefix = ''
        root = f'data/{config.exp_name}'
        dfname = os.path.join(root, prefix + 'edges_date.csv')
        df = pd.read_csv(dfname, header=None)
        p_attrs = np.asarray(loadpkl(os.path.join(root, prefix + 'result_sequence.pkl')))
        pairs_dict = loadpkl(os.path.join(root, prefix + 'pairs.pkl'))
        x_pairs = pairs_dict['x']
        y_pairs = pairs_dict['y']

        x_dataset = OJDataset(df, p_attrs, x_pairs)
        y_dataset = OJDataset(df, p_attrs, y_pairs)

        self.x_loader = DataLoader(x_dataset, config.batch_size,
                                   shuffle=True, pin_memory=True,
                                   num_workers=0)
        self.y_loader = DataLoader(y_dataset, config.batch_size,
                                   shuffle=True, pin_memory=True,
                                   num_workers=0)

        self.n_lnodes = max(df.iloc[:, 0]) + 1
        self.n_rnodes = max(df.iloc[:, 1]) + 1
        self.n_labels = max(df.iloc[:, -1]) + 1
        self.x_freqs = np.zeros(self.n_lnodes, dtype=np.int64)
        self.y_freqs = np.zeros(self.n_rnodes, dtype=np.int64)
        for u, v, *_ in df.itertuples(False):
            self.x_freqs[u] += 1
            self.y_freqs[v] += 1
        config.n_raw_attrs = p_attrs.shape[1]
        config.x_size = self.n_lnodes
        config.y_size = self.n_rnodes
        config.label_size = self.n_labels

        print('Dataset Size: {}'.format(len(self)))
        print('label size:', self.n_labels)

    def __len__(self):
        return max(len(self.x_loader), len(self.y_loader)) * 2

    def iter_batch(self):
        x_iter = iter(self.x_loader)
        y_iter = iter(self.y_loader)
        if len(self.x_loader) < len(self.y_loader):
            x_iter = cycle(x_iter)
        else:
            y_iter = cycle(y_iter)

        for x_batch, y_batch in zip(x_iter, y_iter):
            sx, sy, sa, sl, _, t, ta, _ = x_batch
            yield sx, sy, sa, sl, t, ta, True
            sx, sy, sa, sl, t, _, ta, _ = y_batch
            yield sx, sy, sa, sl, t, ta, False


class OJDataset(Dataset):

    def __init__(self, df, p_attrs, pairs):
        super().__init__()
        self.df = df
        self.p_attrs = p_attrs
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        sx, sy, sp = self.df.iloc[i, :3]
        sa = self.p_attrs[sp]
        sl = self.df.iloc[i, -1]
        tx, ty, tp = self.df.iloc[j, :3]
        ta = self.p_attrs[tp]
        tl = self.df.iloc[j, -1]
        return sx, sy, sa, sl, tx, ty, ta, tl



