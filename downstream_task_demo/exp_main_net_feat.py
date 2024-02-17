import os
import sys
import warnings

import numpy as np
import scanpy as sc
from umap import UMAP
import wandb

from dataloader.stdata import STData
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def svc_train(data, emb, label):
    xgb.set_config(verbosity=3)
    print('Training XGBRegressor')
    bst = XGBRegressor(
        n_jobs=55,
        tree_method = "hist", 
    )
    num_round = 100
    bst.fit(data, emb)
    
    print('Training XGBClassifier')
    clf_svc = XGBClassifier(
        objective='multi:softprob',
        num_class=np.max(label)+1,
        eval_metric='mlogloss',
        nthread=55,
        tree_method="hist",
    )
    
    clf_svc.fit(emb, label)    
    return bst, clf_svc

def svc_pre(input):
    return clf_svc.predict_proba(multioutput_regressor.predict(input))[:,args.aim_label]
    

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
    parser.add_argument('--min_cells', type=int, default=0)

    # Augmentation
    parser.add_argument('--graphwithpca', type=bool, default=True)
    parser.add_argument('--uselabel', type=bool, default=False)
    parser.add_argument('--K_m0', type=int, default=50)
    parser.add_argument('--K_m1', type=int, default=50)

    # Cluster
    parser.add_argument('--cluster_using', type=str, default='gene_rec') # gene_rec
    parser.add_argument('--cluster_method', type=str, default='mclust') # gene_rec
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
    parser.add_argument('--model_dir', type=str, default='test_dlpfc/')
    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--save_dir', type=str, default='exp/')
    parser.add_argument('--plot', type=int, default=0, help='Plot the result')
    parser.add_argument('--var_plot', type=int, default=0)
    parser.add_argument('--plot_louvain', type=int, default=1)
    parser.add_argument('--plot_leiden', type=int, default=0)
    
    parser.add_argument('--num_fea', type=int, default=3000)
    parser.add_argument('--num_sample', type=int, default=1500)
    parser.add_argument('--aim_label', type=int, default=1)
    parser.add_argument('--max_evals', type=int, default=6000)
    
    args = parser.parse_args()

    args.plot = bool(args.plot)
    args.var_plot = bool(args.var_plot)
    args.plot_louvain = bool(args.plot_louvain)
    args.plot_leiden = bool(args.plot_leiden)
    args.cluster_refinement = bool(args.cluster_refinement)
    args.preprocessed = bool(args.preprocessed)
    if args.cluster_method == 'mclust':
        args.cluster_method = 'main'

    if args.run_dir is not None and not os.path.exists(args.run_dir):
        os.mkdir(args.run_dir)
    if args.save_dir is not None:
        args.save_dir += f"{args.dataset}/{args.aim_label}/"
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir+'model/', exist_ok=True)
        with open(args.save_dir + 'setting.txt', 'w') as f:
            f.write(str(args))

    wandb_agent = wandb.init(
        project="MuST_exp_feat",
        entity="liliangyu",
        config=args.__dict__,
        name='EXP-'+''.join(sys.argv[1:]),
        mode=args.wandb,
        save_code=True,
        dir=args.run_dir,
        )
    
    visium = STData(name=args.dataset, crop_size=args.crop_size, bio_norm=False, sample=args.sample)   # Reset sample to get better results.

    adata = visium.adata
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    adata.uns["name"] = args.dataset

    n_clusters = visium.get_annotation_class()
    if n_clusters is not None:
        warnings.warn("n_cluster rewritten due to known label")
    else:
        n_clusters = args.n_clusters

    # clustering & ARI
    full_label = np.load(f'{args.result_dir}{args.dataset}/pred_{args.cluster_method}.npy')
    data = np.load(f'{args.result_dir}{args.dataset}/trans_input.npy')
    emb = np.load(f'{args.result_dir}{args.dataset}/emb.npy')
    hvg = np.load(f'{args.result_dir}{args.dataset}/hvg.npy')
    feature_name = adata.var_names[hvg].tolist()
    
    random_numbers = np.random.choice(len(data), args.num_sample, replace=False)
    
    print('down sample the data')
    data = data[random_numbers]
    emb = emb[random_numbers]
    label = full_label[random_numbers]
    
    print('random select the feature')
    data = data[:, :args.num_fea]
    
    print('explain')
    multioutput_regressor, clf_svc = svc_train(data, emb, label)
    explainer = shap.Explainer(svc_pre, data, feature_names=feature_name[:args.num_fea])
    shap_values = explainer(data, max_evals=args.max_evals)

    plt.figure(figsize=(15, 15))
    shap.plots.heatmap(shap_values, max_display=20, instance_order=shap_values.sum(1))
    plt.tight_layout()
    plt.savefig(args.save_dir + 'shap_heamap.png', dpi=300)
    wandb.log({'shap_heamap': wandb.Image(args.save_dir + 'shap_heamap.png')})

    fig_sp = visium.px_plot_spatial(full_label, background_image=True, save_path=args.save_dir + 'fig_sp.png')
    fig_sp.write_html(args.save_dir + 'fig_sp.html')

    shap_array = shap_values.values
    mean_abs_shap_values = np.mean(np.abs(shap_array), axis=0)
    sorted_indices = np.argsort(-mean_abs_shap_values)
    top_features = [feature_name[i] for i in sorted_indices[:20]]
    print('top_features', top_features)
    for i, gene in enumerate(top_features):
        visium.px_plot_spatial_gene(
            gene, background_image=False, save_path=args.save_dir + f'gene_{gene}.png')
        wandb.log({f'gene_{i}_{gene}': wandb.Image(args.save_dir + f'gene_{gene}.png')})

    with open(f'exp_{args.dataset}.txt', 'a') as f:
        f.write(f'{args.aim_label}: ' + str(top_features) + ',\n')

    wandb.finish()
