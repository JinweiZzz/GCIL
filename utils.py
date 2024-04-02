# -*- coding: utf-8 -*-

import torch
import torch.utils
import torch.utils.data
import numpy as np
import copy
import collections
from sklearn.preprocessing import normalize

from torch_geometric.data import Data
from functools import reduce


# inputted data format: [[treatment, features, outcome], [...], ...]
def create_dataset(test, args, state='ne'):
    outcome = torch.tensor(test[:, -1], dtype=torch.float)
    treatment = torch.tensor(test[:, 0], dtype=torch.float)
    print('treatment', treatment)
    features = torch.tensor(test[:, 1:-1], dtype=torch.float)
    y = outcome.float().view(-1, 1)
    t = treatment.float().view(-1, 1)
    print('y ', y)
    test_dataset = Data(x=features, treatment=treatment, outcome=y, t=t)
    return test_dataset


def split_train_val_test(n, nfolds, seed):
    '''
    n-fold cross validation
    '''
    train_idx, valid_idx = {}, {}
    rnd_state = np.random.RandomState(seed)
    idx_all = rnd_state.permutation(n)
    idx_all = idx_all.tolist()
    stride = int(n / nfolds)
    idx = [idx_all[i * stride:(i + 1) * stride] for i in range(nfolds)]
    for fold in range(nfolds):
        valid_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i != fold:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
    return train_idx, valid_idx


def make_batch(train_len, batch_size, seed):
    """
    return a list of batch ids for mask-based batch.
    Args:
        train_ids: list of train ids
        batch_size: ~
    Output:
        batch ids, e.g., [[1,2,3], [4,5,6], ...]
    """

    num_nodes = train_len
    rnd_state = np.random.RandomState(seed)
    permuted_idx = rnd_state.permutation(num_nodes)
    permuted_train_ids = permuted_idx
    batches = [permuted_train_ids[i * batch_size:(i + 1) * batch_size] for
               i in range(int(num_nodes / batch_size))]
    if num_nodes % batch_size > 0:
        if (num_nodes % batch_size) > 0.5 * batch_size:
            batches.append(
                permuted_train_ids[(num_nodes - num_nodes % batch_size):])
        else:
            batches[-1] = np.concatenate((batches[-1], permuted_train_ids[(num_nodes - num_nodes % batch_size):]))

    return batches, num_nodes


def mean_absolute_percentage_error(y_true, y_pred): # SMAPE
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    return np.mean(2*diff /(abs(y_true)+abs(y_pred)))
