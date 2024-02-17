import torch
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import random
from sklearn.metrics import pairwise_distances

import networkx as nx

from sklearn.cluster import KMeans
from munkres import Munkres

from sklearn.metrics import confusion_matrix

import sys 
sys.path.append("..") 

from utils import targeted_cluster


def Curance_path_list(neighbour_input, distance_input, label):


    row = []
    col = []
    v = []
    n_p, n_n = neighbour_input.shape
    for i in range(n_p):
        for j in range(n_n):
            row.append(i)
            col.append(neighbour_input[i,j])
            v.append(distance_input[i,j])

    G=nx.Graph()
    for i in range(0, n_p):
        G.add_node(i)
    for i in range(len(row)):
        G.add_weighted_edges_from([(row[i],col[i],v[i])])

    # pos=nx.shell_layout(G)
    # nx.draw(G, pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )

    path_list = []
    for i in range(5000):
        source = random.randint(a=0, b=n_p-1)
        source_label = label[source]
        list_with_same_label = np.array(range(n_p))[label==source_label]
        target = random.sample(list(list_with_same_label), 1)

        target = random.randint(a=0, b=n_p-1)
        try:
            path = nx.dijkstra_path(G, source=source, target=target)
            path_list.append(path)
        except:
            pass
    
    return path_list        


class Eval():
    
    def __init__(self, input, latent, label, cuda=None, k=50) -> None:
        n = latent.shape[0]
        if n > 5000:
            random.seed(0)
            index = random.sample(range(n), 5000)
            self.k = k
            # self.k = int(k * 5000/n)
            # print('down sampling')
        else:
            index = range(n)
        self.k = k
        self.input = input.reshape(n, -1)[index]
        self.latent = latent.reshape(n, -1)[index]
        self.label = label[index]
        self.cuda = cuda
        # print('distance_input')
        self.distance_input = self._Distance_squared_CPU(
            self.input, self.input)
        # print('distance_latnet')
        self.distance_latnet = self._Distance_squared_CPU(
            self.latent, self.latent)
        # print('neighbour_input')
        self.neighbour_input, self.rank_input = self._neighbours_and_ranks(self.distance_input)
        # print('neighbour_latent')
        self.neighbour_latent, self.rank_latent = self._neighbours_and_ranks(
            self.distance_latnet)


    def _neighbours_and_ranks(self, distances):
        """
        Inputs:
        - distances,        distance matrix [n times n],
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i)
        """
        k = self.k
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind="stable")

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1 : k + 1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind="stable")
        # print(ranks)

        return neighbourhood, ranks

    
    def _Distance_squared_GPU(self, x, y, cuda=7):

        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(),beta=1, alpha=-2)
        
        d = dist.clamp(min=1e-36)
        return np.sqrt(d.detach().cpu().numpy())

    def _Distance_squared_CPU(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # dist.addmm_(1, -2, x, y.t())
        dist = torch.addmm(dist, mat1=x, mat2=y.t(),beta=1, alpha=-2)
        d = dist.clamp(min=1e-36)
        return d.detach().cpu().numpy()

    def _trustworthiness(self, X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k):
        """
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        """

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(
                Z_neighbourhood[row], X_neighbourhood[row]
            )

            for neighbour in missing_neighbours:
                result += X_ranks[row, neighbour] - k

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result


    def E_Classifacation_SVC(self):

        from sklearn.preprocessing import StandardScaler

        method = SVC(kernel="linear", max_iter=90000)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # if
        n_scores = cross_val_score(
            method, 
            StandardScaler().fit_transform(self.latent),
            self.label.astype(np.int32),
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )

        return n_scores.mean()


    def E_Classifacation_rbfSVC(self):

        from sklearn.preprocessing import StandardScaler

        method = SVC()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # if
        n_scores = cross_val_score(
            method, 
            StandardScaler().fit_transform(self.latent),
            self.label.astype(np.int32),
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )

        return n_scores.mean()

    def E_Curance(self, use_all_data=False):
        # self.E_Curance_pre()

        # if self.label
        # label_ = self.label
        if use_all_data:
            label_ = np.array([0]*self.neighbour_input.shape[0])
        else:
            label_ = self.label
        
        # print(label_)

        path_list = Curance_path_list(self.neighbour_input, self.distance_input, label_)

        # print(path_list)
        alpha_list = []
        for path in path_list:
            if len(path)>3:
                # print(path)
                for i in range(len(path)-3):
                    a_index = path[0]
                    b_index = path[i+1]
                    c_index = path[-1]
                    # print([a_index,b_index,c_index])

                    v1 = self.latent[b_index] - self.latent[a_index]
                    v2 = self.latent[c_index] - self.latent[b_index]
                    cos_alpha = v1.dot(v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
                    alpha = np.arccos(cos_alpha)
                    alpha_list.append(alpha)
        # print( alpha_list )
        # print( alpha_list )
        alpha_list = np.array(alpha_list)
        alpha_list = alpha_list[~np.isnan(alpha_list)]
        return np.mean( alpha_list )
    
    
    def E_Curance_2(self, use_all_data=False):
        # self.E_Curance_pre()

        # if self.label
        # label_ = self.label
        if use_all_data:
            label_ = np.array([0]*self.neighbour_input.shape[0])
        else:
            label_ = self.label
        
        # print(label_)

        path_list = Curance_path_list(self.neighbour_input, self.distance_input, label_)

        # print(path_list)
        alpha_list = []
        for path in path_list:
            if len(path)>3:
                # print(path)
                for i in range(len(path)-3):
                    a_index = path[i]
                    b_index = path[i+1]
                    c_index = path[i+2]

                    v1 = self.latent[b_index] - self.latent[a_index]
                    v2 = self.latent[c_index] - self.latent[b_index]
                    cos_alpha = v1.dot(v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
                    alpha = np.arccos(cos_alpha)
                    alpha_list.append(alpha)
        # print( alpha_list )
        # print( alpha_list )
        alpha_list = np.array(alpha_list)
        alpha_list = alpha_list[~np.isnan(alpha_list)]
        return np.mean( alpha_list )

    def TestClassifacationKMeans(self, embedding, label, n_clusters=None):


        # l1 = list(set(label))
        # numclass1 = len(l1)
        # predict_labels = KMeans(n_clusters=numclass1, random_state=0).fit_predict(embedding)

        # l2 = list(set(predict_labels))
        # numclass2 = len(l2)

        # cost = np.zeros((numclass1, numclass2), dtype=int)
        # for i, c1 in enumerate(l1):
        #     mps = [i1 for i1, e1 in enumerate(label) if e1 == c1]
        #     for j, c2 in enumerate(l2):
        #         mps_d = [i1 for i1 in mps if predict_labels[i1] == c2]
        #         cost[i][j] = len(mps_d)

        # # match two clustering results by Munkres algorithm
        # m = Munkres()
        # cost = cost.__neg__().tolist()

        # indexes = m.compute(cost)

        # # get the match results
        # new_predict = np.zeros(len(predict_labels))
        # for i, c in enumerate(l1):
        #     # correponding label in l2:
        #     c2 = l2[indexes[i][1]]

        #     # ai is the index with label==c2 in the pred_label list
        #     ai = [ind for ind, elm in enumerate(predict_labels) if elm == c2]
        #     new_predict[ai] = int(c)

        # acc = metrics.accuracy_score(label, new_predict)
        
        true = label
        class_num = len(np.unique(true))
        # pred = SpectralClustering(n_clusters=class_num, assign_labels='discretize').fit_predict(ins_emb)
        pred = KMeans(n_clusters=class_num if n_clusters is None else n_clusters, random_state=0).fit_predict(embedding)

        cnt = len(true)
        cm = cnt - confusion_matrix(pred, true)
        idxs = Munkres().compute(cm)
        idxs = dict(idxs)
        for i, num in enumerate(pred):
            pred[i] = idxs[num]

        self.k_means_pre = pred
        
        acc = metrics.accuracy_score(label, pred)

        return acc #, nmi, f1_macro, precision_macro, adjscore


    def E_Clasting_Kmeans(self, n_clusters=None):

        # from sklearn.preprocessing import StandardScaler

        # method = SVC(kernel="linear", max_iter=90000)
        # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # if
        # n_scores = cross_val_score(
        #     method, 
        #     StandardScaler().fit_transform(self.latent),
        #     self.label.astype(np.int32),
        #     scoring="accuracy",
        #     cv=cv,
        #     n_jobs=-1
        # )
        return self.TestClassifacationKMeans(self.latent, self.label.astype(np.int32), n_clusters=n_clusters)

    def E_Clasting_louvain(self, n_clusters=None):

        true_label = self.label.astype(np.int32)
        embedding = self.latent
        class_num = np.max(true_label) + 1
        # pred = SpectralClustering(n_clusters=class_num, assign_labels='discretize').fit_predict(ins_emb)
        # pred = KMeans(n_clusters=class_num if n_clusters is None else n_clusters, random_state=0).fit_predict(embedding)
        pred = targeted_cluster(embedding, target_n_clusters=11)

        cnt = len(true_label)
        cm = cnt - confusion_matrix(pred, true_label)
        idxs = Munkres().compute(cm)
        idxs = dict(idxs)
        for i, num in enumerate(pred):
            pred[i] = idxs[num]

        self.louvain_pre = pred
        
        acc = metrics.accuracy_score(true_label, pred)

        return acc #, nmi, f1_macro, precision_macro, adjscore

    def E_Classifacation_KNN(self):

        from sklearn.neighbors import KNeighborsClassifier
        method = KNeighborsClassifier(n_neighbors=3)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # if
        n_scores = cross_val_score(
            method, self.latent, self.label.astype(np.int32), scoring="accuracy", cv=cv, n_jobs=-1
        )

        return n_scores.mean()

    def E_NNACC(self):

        indexNN = self.neighbour_latent[:, 0].reshape(-1)
        labelNN = self.label[indexNN]
        acc = (self.label == labelNN).sum() / self.label.shape[0]

        return acc

    def E_mrre(self, ):
        """
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        """
        
        k=self.k

        X_neighbourhood, X_ranks = self.neighbour_input, self.rank_input
        Z_neighbourhood, Z_ranks = self.neighbour_latent, self.rank_latent

        n = self.distance_input.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx - rz) / rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx - rz) / rx

        # Normalisation constant
        C = n * sum([abs(2 * j - n - 1) / j for j in range(1, k + 1)])
        return mrre_ZX / C, mrre_XZ / C
    
    def E_distanceAUC(self,):

        disZN = (self.distance_latnet-self.distance_latnet.min())/(self.distance_latnet.max()-self.distance_latnet.min())
        LRepeat = self.label.reshape(1,-1).repeat(self.distance_latnet.shape[0], axis=0)
        L = (LRepeat==LRepeat.T).reshape(-1)
        auc = metrics.roc_auc_score(1-L, disZN.reshape(-1))
        
        return auc

    def E_trustworthiness(self):
        X_neighbourhood, X_ranks = self.neighbour_input, self.rank_input
        Z_neighbourhood, Z_ranks = self.neighbour_latent, self.rank_latent
        n = self.distance_input.shape[0]
        return self._trustworthiness(
            X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, self.k
        )

    def E_continuity(self):
        """
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        """

        X_neighbourhood, X_ranks = self.neighbour_input, self.rank_input
        Z_neighbourhood, Z_ranks = self.neighbour_latent, self.rank_latent
        n = self.distance_input.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(
            Z_neighbourhood, Z_ranks, X_neighbourhood, X_ranks, n, self.k
        )

    def E_Rscore(self):
        # n = self.distance_input.shape[0]
        import scipy
        r = scipy.stats.pearsonr(self.distance_input.reshape(-1), self.distance_latnet.reshape(-1))
        # print(r)
        return r[0]

    def E_Dismatcher(self):   
        emb, label = self.latent, self.label
        list_dis = []
        for i in list(set(label)):
            p = emb[label==i]
            m = p.mean(axis=0)[None,:]
            list_dis.append(pairwise_distances(p, m).mean())
        list_dis = np.array(list_dis)
        list_dis_norm=list_dis/list_dis.max()        
        sort1 = np.argsort(list_dis_norm)
        # print('latent std:', list_dis_norm)
        # print('latent sort:', sort1)

        emb, label = self.input, self.label
        emb = emb.reshape(emb.shape[0],-1)
        list_dis = []
        for i in list(set(label)):
            p = emb[label==i]
            m = p.mean(axis=0)[None,:]
            list_dis.append(pairwise_distances(p, m).mean())
        list_dis = np.array(list_dis)
        list_dis_norm=list_dis/list_dis.max()        
        sort2 = np.argsort(list_dis_norm)
        # print('latent std:', list_dis_norm)
        # print('latent sort:', sort2)


        v, s, t = 0, sort2.tolist(), sort1.tolist()  
        for i in range(len(t)):
            if t[i] != s[i]:
                v = v + abs(t.index(s[i])-i)
        s_constant = (2.0/len(s)**2)

        return v * s_constant