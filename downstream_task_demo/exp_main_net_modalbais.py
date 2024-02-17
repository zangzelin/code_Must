import os
import sys
import warnings

import numpy as np
import wandb

from dataloader.stdata import STData
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def svc_train(data, emb, label):
    print('Training XGBClassifier')
    clf_svc = XGBClassifier(
        objective='multi:softprob',
        num_class=np.max(label)+1,
        eval_metric='mlogloss',
        nthread=55,
        tree_method="hist",
    )
    
    clf_svc.fit(data, label)    
    return clf_svc



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R_HOME', type=str, default='/root/miniconda3/lib/R')
    parser.add_argument("--wandb", type=str, default="online")

    # Datasets
    parser.add_argument('--dataset', type=str, default='151673')
    parser.add_argument('--sample', type=str, default='barcode')
    parser.add_argument('--n_top_genes', type=int, default=3000)
    parser.add_argument('--max_value', type=int, default=10)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--preprocessed', type=int, default=0)
    parser.add_argument('--min_cells', type=int, default=50)

    # Augmentation
    parser.add_argument('--graphwithpca', type=bool, default=True)
    parser.add_argument('--uselabel', type=bool, default=False)
    parser.add_argument('--K_m0', type=int, default=50)
    parser.add_argument('--K_m1', type=int, default=50)

    # Cluster
    parser.add_argument('--cluster_using', type=str, default='gene_rec') # gene_rec
    parser.add_argument('--n_clusters', type=int, default=15)
    parser.add_argument('--radius', type=int, default=50)
    parser.add_argument('--cluster_refinement', type=int, default=0)
    
    # Model
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--dim_input', type=int, default=3000)
    parser.add_argument('--dim_output', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--aug_rate_0', type=float, default=0.1)
    parser.add_argument('--aug_rate_1', type=float, default=0.1)
    parser.add_argument('--v_latent', type=float, default=0.01)
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--n_encoder_layer', type=int, default=1, help='number of encoder layers')
    parser.add_argument('--n_fusion_layer', type=int, default=1)
    parser.add_argument('--bn_type', type=str, default='bn')
    parser.add_argument('--self_loop', type=int, default=0)
    parser.add_argument('--down_sample_rate', type=float, default=1)
    parser.add_argument('--morph_trans_ratio', type=float, default=1)
    parser.add_argument('--aug_method', type=str, default="near_mix")
    parser.add_argument('--run_dir', type=str, default=os.getenv('WANDB_RUN_DIR'))
    parser.add_argument('--model_dir', type=str, default='model/')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--plot', type=int, default=0, help='Plot the result')
    parser.add_argument('--var_plot', type=int, default=0)
    parser.add_argument('--plot_louvain', type=int, default=1)
    parser.add_argument('--plot_leiden', type=int, default=0)
    
    parser.add_argument('--norm', type=str, default="none")
    parser.add_argument('--num_fea', type=int, default=3000)
    parser.add_argument('--num_sample', type=int, default=1500)
    parser.add_argument('--aim_label', type=int, default=1)
    parser.add_argument('--max_evals', type=int, default=5500)
    
    args = parser.parse_args()

    args.plot = bool(args.plot)
    args.var_plot = bool(args.var_plot)
    args.plot_louvain = bool(args.plot_louvain)
    args.plot_leiden = bool(args.plot_leiden)
    args.cluster_refinement = bool(args.cluster_refinement)
    args.preprocessed = bool(args.preprocessed)

    if args.run_dir is not None and not os.path.exists(args.run_dir):
        os.mkdir(args.run_dir)
    if args.save_dir is not None:
        args.save_dir += f"{args.dataset}/"
        os.makedirs(args.save_dir, exist_ok=True)
    
    wandb.init(
        project="MuST_modalbias", 
        name=f'dataset{args.dataset}_num_fea{args.num_fea}_num_sample{args.num_sample}_aim_label{args.aim_label}_max_evals{args.max_evals}_sample{args.sample}',
        config=args,
        )
    
    visium = STData(name=args.dataset, crop_size=args.crop_size, bio_norm=False, sample=args.sample)   # Reset sample to get better results.

    adata = visium.adata
    adata.uns["name"] = args.dataset

    
    emb = np.load(f'results/{args.dataset}/emb.npy')
    label = np.load(f'results/{args.dataset}/pred_main.npy')
    trans_dict = {v:i for i, v in enumerate(np.unique(label))}
    label = np.vectorize(trans_dict.__getitem__)(label)
    trans = np.load(f'results/{args.dataset}/trans_input.npy')
    hvg = np.load(f'results/{args.dataset}/hvg.npy')
    adata = adata[:, hvg]
    morph = visium.get_morph()
    loc = visium.get_coords()
    loc = np.concatenate([loc, (loc[:, 0] + loc[:, 1]).reshape(-1, 1), (loc[:, 0] - loc[:, 1]).reshape(-1, 1)], axis=1)
    if args.norm == 'std':
        loc = StandardScaler().fit_transform(loc)
    elif args.norm == 'minmax':
        loc = MinMaxScaler().fit_transform(loc)
    data = np.concatenate([trans, morph, loc], axis=1)
    
    feature_name = adata.var_names.tolist()
    
    print('explain')
    clf_svc = svc_train(data, emb, label)
    explainer = shap.Explainer(clf_svc, data, feature_names=feature_name[:args.num_fea])
    shap_values = explainer(data)
    
    shap_values = np.abs(shap_values.values)

    imp_mor = shap_values[:, trans.shape[1] + morph.shape[1]:].max((1,2))
    imp_gen = shap_values[:, :trans.shape[1] + morph.shape[1]].max((1,2))
    
    mor_rate = imp_mor / (imp_mor + imp_gen)
    save_dir = 'res_multimodal/'
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir + f'mor_rate_{args.dataset}_{args.norm}', mor_rate)
    fig = visium.px_plot_spatial_gene(mor_rate, background_image=False, dpi=400, save_path=save_dir + f'rate_{args.dataset}_{args.norm}.png')
    fig = visium.px_plot_spatial_gene(mor_rate, background_image=True, dpi=400, save_path=save_dir + f'rate_{args.dataset}_{args.norm}_bg.png')
    colors = ['#2E91E5', '#E15F99', '#1CA71C']
    plt.figure(figsize=(3, 4), dpi=400)
    sns.violinplot(mor_rate, color=colors[0])
    plt.xticks([])
    plt.savefig(save_dir + f'violin_{args.dataset}_{args.norm}.png')
    wandb.log({'rate': fig})

    wandb.finish()