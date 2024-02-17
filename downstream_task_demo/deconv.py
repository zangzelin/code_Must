import os
import wandb
import seaborn
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

from dataloader.stdata import STData

def Analysis_mix(label, emb):
    cell_mean_value_list = []
    for c in range(np.max(label)+1):
        cell_mean_value_list.append(np.mean(emb[label==c], axis=0))

    cell_mean_value_numpy = np.array(cell_mean_value_list)
    dis = []
    for i in range(emb.shape[0]):
        label_c = label[i]
        dis.append(np.sqrt(
            np.sum((cell_mean_value_numpy[label_c] - emb[i])**2)
            ))
    dis = np.array(dis)
    
    return dis, cell_mean_value_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_fea', type=int, default=300)
    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--save_dir', type=str, default='deconv/')
    parser.add_argument('--cluster_method', type=str, default='mclust')
    parser.add_argument('--dataset', type=str, default='200115_08')
    parser.add_argument('--min_cells', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()

    args.save_dir += args.dataset + '/'
    os.makedirs(args.save_dir, exist_ok=True)
    if args.cluster_method == 'mclust':
        args.cluster_method = 'main'

    wandb.init(
        project="MuST_deconv", 
        entity="liliangyu",
        name=f'Deconv_{args.dataset}',
        config=args.__dict__,
        save_code=True,
    )

    visium = STData(name=args.dataset, bio_norm=False)
    adata = visium.adata
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    data = np.load(f"{args.result_dir}{args.dataset}/trans_input.npy")[:,:args.num_fea]
    emb = np.load(f"{args.result_dir}{args.dataset}/emb.npy")
    label = np.load(f'{args.result_dir}{args.dataset}/pred_{args.cluster_method}.npy')
    hvg = np.load(f'{args.result_dir}{args.dataset}/hvg.npy')
    
    print('dataset loaded')

    feature_name = adata.var_names[hvg].tolist()[:args.num_fea]

    dis, cell_mean_value_list = Analysis_mix(label, emb)

    weight_list = []
    for i in range(emb.shape[0]):
        A = np.array(cell_mean_value_list).T  # (20, 72)
        B = emb[i] # (1, 72)
        model = Lasso(alpha=args.alpha)
        model.fit(A,B)
        weight = model.coef_
        weight_list.append(weight)
        
    weight = np.array(weight_list).T
    weight[weight<1e-5] = 0
    w_sum = weight.sum(axis=0)
    n_weight = weight / w_sum
    np.save(args.save_dir+'val.npy', weight)
    np.save(args.save_dir+'n_val.npy', n_weight)
    
    fig = plt.figure(figsize=(20,5), dpi=300)
    seaborn.heatmap(weight, cmap='Reds',)
    plt.tight_layout()

    fig_sp = visium.px_plot_spatial(label, background_image=True, save_path=args.save_dir + 'fig_sp.png')
    fig_sp.write_html(args.save_dir + 'fig_sp.html')
    for i in np.unique(label):
        visium.px_plot_spatial_gene(weight[i], save_path=args.save_dir + f'sp_{i}.png')
        visium.px_plot_spatial_gene(n_weight[i], save_path=args.save_dir + f'spn_{i}.png')

        fig = plt.figure(figsize=(20,5), dpi=300)
        plt.title(f'Cluster {i}')
        seaborn.heatmap(weight[:, label==i], cmap='Reds',)
        plt.tight_layout()
        plt.savefig(args.save_dir + f'hm_{i}.png', dpi=200)
        plt.close()
        
        fig = plt.figure(figsize=(20,5), dpi=300)
        plt.title(f'Cluster {i}')
        seaborn.heatmap(n_weight[:, label==i], cmap='Reds',)
        plt.tight_layout()
        plt.savefig(args.save_dir + f'hmn_{i}.png', dpi=200)
        plt.close()