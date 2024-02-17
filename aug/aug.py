import os
import joblib
import logging

import torch
import numpy as np
from sklearn.metrics import pairwise_distances

def aug_near_mix(index, dataset, neighbors_index, k=10, random_t=0.1, device="cuda", ):
    r = (
        torch.arange(start=0, end=index.shape[0]) * k
        + torch.randint(low=1, high=k, size=(index.shape[0],))
    ).to(device)
    random_select_near_index = (
        neighbors_index[index][:, :k].reshape((-1,))[r].long()
    )
    random_select_near_data2 = dataset.data[random_select_near_index]
    random_rate = torch.rand(size=(index.shape[0], 1)).to(device) * random_t
    data_cuda_index = dataset.data[index].to(device)
    random_select_near_data2 = random_select_near_data2.to(device)

    return (
        random_rate * random_select_near_data2 + (1 - random_rate) * data_cuda_index
    )


def aug_near_feautee_change(index, dataset, neighbors_index, k=10, random_t=0.99, device="cuda"):
    r = torch.arange(start=0, end=index.shape[0], device=device) * k + torch.randint(low=1, high=k, size=(index.shape[0],), device=device)
    
    random_select_near_index = (
        neighbors_index[index][:, :k].reshape((-1,))[r].long()
    )
    random_select_near_data2 = dataset[random_select_near_index]
    data_origin = dataset[index]
    random_rate = torch.rand(size=(1, data_origin.shape[1]), device=device)
    random_mask = (random_rate > random_t).reshape(-1).float()
    return random_select_near_data2 * random_mask + data_origin * (1 - random_mask)


def aug_randn(index, dataset, neighbors_index=None, k=10, random_t=0.01, device="cuda"):
    data_origin = dataset[index]
    return (
        data_origin
        + torch.randn(data_origin.shape, device=data_origin.device) * torch.var(dataset, dim=0) * random_t
    )

def cal_near_index(data, label=None, k=10, device="cuda", uselabel=False, modal=None, graphwithpca=False, dataset="placeholder", unique_str=""):
    filename = f"save_near_index/pca{graphwithpca}dataset{dataset}K{k}uselabel{uselabel}modal{modal}n{data.shape[0]}w{data.shape[1]}{unique_str}"

    os.makedirs("save_near_index", exist_ok=True)
    if not os.path.exists(filename):
        X_rshaped = (
            data.reshape(
                (data.shape[0], -1)).detach().cpu().numpy()
        )
        # if graphwithpca and X_rshaped.shape[1]>50:
        #     X_rshaped = PCA(n_components=50).fit_transform(X_rshaped)
        if not uselabel:
            # index = NNDescent(X_rshaped, n_jobs=-1)
            dis = pairwise_distances(X_rshaped)
            neighbors_index = dis.argsort(axis=1)[:, 1:k + 1]
            # print('X_rshaped', X_rshaped)
            # print('neighbors_index', neighbors_index)
            # neighbors_index, neighbors_dist = index.query(X_rshaped, k=k+1)
            # neighbors_index = neighbors_index[:,1:]
        else:
            dis = pairwise_distances(X_rshaped)
            M = np.repeat(label.reshape(1, -1), X_rshaped.shape[0], axis=0)
            dis[(M - M.T) != 0] = dis.max() + 1
            neighbors_index = dis.argsort(axis=1)[:, 1:k + 1]
        joblib.dump(value=neighbors_index, filename=filename)

        logging.debug(f"save data to {filename}")
    else:
        logging.debug(f"load data from {filename}")
        neighbors_index = joblib.load(filename)

    return torch.tensor(neighbors_index).to(device)