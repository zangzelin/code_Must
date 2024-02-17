import torch
from preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, get_feature, permutation, fix_seed
import time
import random
import logging
import warnings
import numpy as np
from model import Encoder_MUST
from aug import aug
from tqdm import tqdm, trange
from torch import nn
import torch.nn.functional as F
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import scipy.sparse as sp


import pandas as pd
import wandb

class MUST():
    def __init__(self,
                 adata,
                 morph,
                 n_top_genes=3000,
                 max_value=1,
                 adata_sc=None,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 random_seed=41,
                 alpha=1,
                 beta=1,
                 theta=0.1,
                 v_latent=0.01,
                 datatype='10X',
                 aug_rate_0=0.1,
                 aug_rate_1=0.1,
                 n_encoder_layer=1,
                 n_fusion_layer=1,
                 bn_type='bn', 
                 self_loop=1,
                 morph_trans_ratio=0.5,
                 graphwithpca=False,
                 uselabel=False,
                 K_m0=5,
                 K_m1=5,
                 aug_method="randn",
                 unique_str="",
                 preprocessed=False,
                 down_sample_rate=0.1,
                 min_cells=50,
                 ):
        self.adata = adata.copy()
        self.morph = morph
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.datatype = datatype
        self.v_latent = v_latent
        self.v_input = 100
        self.aug_rate_0 = aug_rate_0
        self.aug_rate_1 = aug_rate_1
        self.n_encoder_layer = n_encoder_layer
        self.n_fusion_layer = n_fusion_layer
        self.bn_type = bn_type
        self.self_loop = self_loop
        self.morph_trans_ratio = morph_trans_ratio
        self.graphwithpca=graphwithpca
        self.uselabel=uselabel
        self.K_m0=K_m0
        self.K_m1=K_m1
        self.aug_method=aug_method
        self.unique_str=unique_str
        self.down_sample_rate=down_sample_rate

        self.dataset = adata.uns["name"]

        fix_seed(self.random_seed)

        if not preprocessed and 'highly_variable' not in adata.var.keys():
            if self.datatype == '10x':
                self.adata = preprocess(self.adata, n_top_genes=n_top_genes, max_value=max_value, dataset=self.dataset)
            else:
                self.adata = preprocess(self.adata, min_cells=min_cells, n_top_genes=n_top_genes, max_value=max_value, dataset=self.dataset)
                
        fix_seed(self.random_seed)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['stereo', 'slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(
            self.adata.obsm['feat'].copy()).to(self.device)
        print(self.features)
        self.features_a = torch.FloatTensor(
            self.adata.obsm['feat_a'].copy()).to(self.device)
        self.morph = morph
        if self.morph is not None:
            self.morph = torch.FloatTensor(
                morph).to(self.device)
            self.morph_a = torch.FloatTensor(
                permutation(morph)).to(self.device)
        self.adj = self.adata.obsm['adj'] + np.eye(self.adata.obsm['adj'].shape[0]) * self.self_loop

        self.graph_neigh = torch.FloatTensor(
            self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0]) * self.self_loop).to(self.device)
        self.neighbor_index_a = aug.cal_near_index(data=self.features, k=K_m1, uselabel=uselabel, graphwithpca=graphwithpca, device=self.device, modal='0', dataset=self.dataset, unique_str=unique_str)
        if self.morph is not None:
            self.neighbor_index_b = aug.cal_near_index(data=self.morph, k=K_m0, uselabel=uselabel, graphwithpca=graphwithpca, device=self.device, modal='1', dataset=self.dataset, unique_str=unique_str)

        if self.morph is not None:
            self.input_morph = morph
        self.input_trans = self.adata.obsm['feat']

        self.dim_input_a = self.features.shape[1]
        if self.morph is not None:
            self.dim_input_b = self.morph.shape[1]
        else:
            self.dim_input_b = None
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            # standard version
            self.adj = preprocess_adj(self.adj)
            self.adj_coo = sp.coo_matrix(self.adj)

            indices = torch.LongTensor([self.adj_coo.row, self.adj_coo.col])
            values = torch.FloatTensor(self.adj_coo.data)
            shape = self.adj_coo.shape
            
            self.adj_sparse = torch.sparse_coo_tensor(indices, values, shape).to(self.device)


    def _TwowaydivergenceLoss(self, P_, Q_, select=None):

        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)

        return losssum.mean()

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0]) == 1] = 1e-12

        if metric == "cossim":
            input_a, input_b = x, x
            normalized_input_a = torch.nn.functional.normalize(input_a)
            normalized_input_b = torch.nn.functional.normalize(input_b)
            dist = torch.mm(normalized_input_a, normalized_input_b.T)
            dist *= -1
            dist += 1

            dist[torch.eye(dist.shape[0]) == 1] = 1e-12

        return dist

    def _CalGamma(self, v):

        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

    def _Similarity(self,
                    dist,
                    gamma,
                    v):
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        Pij = (
            gamma
            * torch.tensor(2 * 3.14)
            * gamma
            * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij

    def loss_manifold(
        self,
        input_data,
        latent_data,
        v_latent,
        metric='euclidean',
    ):

        data_1 = input_data[: input_data.shape[0] // 2]

        dis_P = self._DistanceSquared(data_1, metric=metric)
        latent_data_1 = latent_data[: input_data.shape[0] // 2]

        dis_P_2 = dis_P  # + nndistance.reshape(1, -1)
        P_2 = self._Similarity(dist=dis_P_2,
                               gamma=self._CalGamma(self.v_input),
                               v=self.v_input, )
        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        loss_ce_2 = self._TwowaydivergenceLoss(P_=P_2, Q_=Q_2)
        return loss_ce_2

    def augmentation(self, fea, t=0.1):
        fea_rand = torch.randn(fea.shape, device=fea.device) * torch.var(fea, dim=0) * t
        return fea + fea_rand

    def train(self, verbose=True):
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(
                self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            #    self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
            self.model = Encoder_MUST(
                self.dim_input_a,
                self.dim_input_b,
                self.dim_output,
                self.graph_neigh,
                n_encoder_layer=self.n_encoder_layer,
                n_fusion_layer=self.n_fusion_layer,
                bn_type=self.bn_type,
                morph_trans_ratio=self.morph_trans_ratio,
                platform=self.datatype,
                ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        logging.info('Begin to train ST data...')
        self.model.train()
        self.adata.obsm['trans_input'] = self.features.detach().cpu().numpy()

        tmp_idx = torch.tensor(np.arange(self.features.shape[0]), dtype=int) # As augmentation index when batch is not adopted.
        if verbose:
            tr = trange(self.epochs)
        else:
            tr = range(self.epochs)
        for epoch in tr:
            self.model.train()

            aug_func = getattr(aug, f"aug_{self.aug_method}")
            
            if self.morph is not None:
                self.morph_a = aug_func(index=tmp_idx, dataset=self.morph, neighbors_index=self.neighbor_index_b,
                                            k=self.K_m0, random_t=self.aug_rate_0, device=self.device)
            else:
                self.morph_a = None
            self.features_a = aug_func(index=tmp_idx, dataset=self.features, neighbors_index=self.neighbor_index_a,
                                           k=self.K_m1, random_t=self.aug_rate_1, device=self.device)
            hiden_feat_list, self.emb, __, __ = self.model(
                self.features, self.morph, self.adj_sparse)
            hiden_feat_list_a, self.emb_a, __, __ = self.model(
                self.features_a, self.morph_a, self.adj_sparse)

            [self.hiden_feat, self.hiden_feat_p] = hiden_feat_list
            [self.hiden_feat_a, self.hiden_feat_p_a] = hiden_feat_list_a

            down_sample_mask = torch.rand(self.hiden_feat.shape[0]) < self.down_sample_rate
            self.d_hiden_feat = self.hiden_feat[down_sample_mask]
            self.d_hiden_feat_p = self.hiden_feat_p[down_sample_mask]
            self.d_hiden_feat_a = self.hiden_feat_a[down_sample_mask]
            self.d_hiden_feat_p_a = self.hiden_feat_p_a[down_sample_mask]
            self.d_emb = self.emb[down_sample_mask]
            self.d_emb_a = self.emb_a[down_sample_mask]
            self.d_features = self.features[down_sample_mask]

            self.man_loss = self.loss_manifold(
                input_data=torch.cat([self.d_hiden_feat, self.d_hiden_feat_a], dim=0),
                latent_data=torch.cat([self.d_hiden_feat_p, self.d_hiden_feat_p_a], dim=0),
                v_latent=self.v_latent,
            )
            # self.man_loss = 0
            self.feat_loss = F.mse_loss(self.d_features, self.d_emb)
            # self.feat_loss = torch.mean(self.d_features)

            loss = self.alpha*self.feat_loss + self.beta*self.man_loss
            # loss = self.alpha*self.feat_loss
            # loss = self.beta*self.man_loss

            wandb.log({'feat_loss': self.alpha*self.feat_loss,
                       'man_loss': self.beta*self.man_loss,
                       'all_loss': loss})

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logging.info("Optimization finished for ST data!")

        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(
                    self.features, self.morph, self.adj_sparse)[1]
                self.emb_rec = F.normalize(
                    self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            else:
                raw_output = self.model(self.features, self.morph, self.adj_sparse)
                self.lat = raw_output[0][1].detach().cpu().numpy()
                self.rec = raw_output[1].detach().cpu().numpy()
                self.trans_emb = raw_output[2].detach().cpu().numpy()
                self.morph_emb = raw_output[3].detach().cpu().numpy() if raw_output[3] is not None else None
            self.adata.obsm['emb'] = self.lat
            self.adata.obsm['gene_rec'] = self.rec

            return self.adata
            
    def discover_region(self):
        raw_output = self.model(self.features, self.morph, self.adj_sparse)
        self.lat = raw_output[0][1].detach().cpu().numpy()
        self.rec = raw_output[1].detach().cpu().numpy()
        self.trans_emb = raw_output[2].detach().cpu().numpy()
        self.morph_emb = raw_output[3].detach().cpu().numpy() if raw_output[3] is not None else None
        self.adata.obsm['emb'] = self.lat
        self.adata.obsm['gene_rec'] = self.rec

        return self.adata

    def save(self, save_dir=''):
        self.model.save(save_dir)

    def load(self, load_dir=''):
        self.model = Encoder_MUST(
            self.dim_input_a,
            self.dim_input_b,
            self.dim_output,
            self.graph_neigh,
            n_encoder_layer=self.n_encoder_layer,
            n_fusion_layer=self.n_fusion_layer,
            bn_type=self.bn_type,
            morph_trans_ratio=self.morph_trans_ratio,
            platform=self.datatype,
            ).to(self.device)
        self.model.load(load_dir)