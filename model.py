import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_MUST(Module):
    def __init__(self, in_features_a, in_features_b, out_features, graph_neigh, n_encoder_layer=1, dropout=0.0, act=F.relu, morph_trans_ratio=0.5, platform='10x',
                 bn_type='bn', n_fusion_layer=1):
        super(Encoder_MUST, self).__init__()
        self.in_features_a = in_features_a
        self.in_features_b = in_features_b
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.morph_trans_ratio = morph_trans_ratio
        self.platform=platform
        
        self.encoder_a = nn.Sequential()
        for i in range(n_encoder_layer):
            if i == 0:
                self.encoder_a.append(nn.Linear(self.out_features, self.in_features_a))
            else:
                self.encoder_a.append(nn.Linear(self.out_features, self.out_features))
        if self.in_features_b is not None:
            self.encoder_b = nn.Sequential()
            for i in range(n_encoder_layer):
                if i == 0:
                    self.encoder_b.append(nn.Linear(self.out_features, self.in_features_b))
                else:
                    self.encoder_b.append(nn.Linear(self.out_features, self.out_features))
        self.mlp_out = Parameter(torch.FloatTensor(self.out_features, self.in_features_a))
        self.reset_parameters()
        
        if bn_type == 'bn':
            self.batch_norm = nn.BatchNorm1d(out_features)
        elif bn_type == 'none':
            self.batch_norm = nn.Identity(out_features)

        if n_fusion_layer == 1:
            self.mlp = nn.Linear(self.out_features, self.out_features,)
        else:
            self.mlp = nn.Sequential()
            for i in range(n_fusion_layer):
                self.mlp.append(nn.Linear(self.out_features, self.out_features))

        self.sigm = nn.Sigmoid()
        print(self.encoder_a)
        print(self.mlp)
        print(self.batch_norm)
        
    def reset_parameters(self):
        for weight in self.encoder_a:
            torch.nn.init.xavier_uniform_(weight.weight)
        if self.in_features_b is not None:
            for weight in self.encoder_b:
                torch.nn.init.constant_(weight.weight, 0)
        torch.nn.init.xavier_uniform_(self.mlp_out)

    def head_fwd(self, encoder, data, adj):
        for i, weight in enumerate(encoder):
            if i == 0:
                z = F.dropout(data, self.dropout, self.training)
                z = torch.mm(z, weight.weight)
                z = torch.mm(adj, z)
            else:
                z = F.dropout(z, self.dropout, self.training)
                z = torch.mm(z, weight.weight)
                z = torch.mm(adj, z)

        return z

    def forward(self, feat_a, feat_b, adj):
        z1 = self.head_fwd(self.encoder_a, feat_a, adj)
        z2 = None
        if feat_b is not None:
            z2 = self.head_fwd(self.encoder_b, feat_b, adj)
        
        # hiden_emb = torch.concat([z1, z2], axis=1)
        if feat_b is not None:
            # import numpy as np
            # np.save('trans_emb.npy', z1.detach().cpu().numpy())
            # np.save('morph_emb.npy', z2.detach().cpu().numpy())
            # import pdb;pdb.set_trace()
            hiden_emb =  z1 * self.morph_trans_ratio + z2 * (1 - self.morph_trans_ratio)    # z1 trans z2 morph
        else:
            hiden_emb = z1
        hiden_emb2 = self.mlp(hiden_emb)
        hiden_emb2 = self.batch_norm(hiden_emb2)
        
        h = torch.mm(hiden_emb2, self.mlp_out)
        h = torch.sparse.mm(adj, h)
        
        return [hiden_emb, hiden_emb2], h, z1, z2
    
    def save(self, save_dir=''):
        torch.save(self.encoder_a, save_dir + 'encoder_a.pt')
        torch.save(self.mlp, save_dir + 'mlp.pt')
        torch.save(self.mlp_out, save_dir + 'mlp_out.pt')
        if self.in_features_b is not None:
            torch.save(self.encoder_b, save_dir + 'encoder_b.pt')

    def load(self, load_dir=''):
        self.encoder_a = torch.load(load_dir + 'encoder_a.pt')
        self.mlp = torch.load(load_dir + 'mlp.pt')
        self.mlp_out = torch.load(load_dir + 'mlp_out.pt')
        if self.platform == '10x':
            self.encoder_b = torch.load(load_dir + 'encoder_b.pt')
