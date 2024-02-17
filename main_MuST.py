import os
import sys
import warnings

import wandb
import numpy as np
from umap import UMAP

import torch
from MUST import MUST
from dataloader.stdata import STData

import eval.eval_core_base as ecb
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from utils import cluster, refine_label, make_error_label, targeted_cluster, cluster_map, aligned_accuracy_score, stable_cluster

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R_HOME', type=str, default='/root/miniconda3/lib/R')
    parser.add_argument("--wandb", type=str, default="online")

    # Datasets
    parser.add_argument('--dataset', type=str, default='V1_Adult_Mouse_Brain')
    parser.add_argument('--sample', type=str, default='barcode')
    parser.add_argument('--n_top_genes', type=int, default=3000)
    parser.add_argument('--max_value', type=int, default=10)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--preprocessed', type=int, default=0)
    parser.add_argument('--min_cells', type=int, default=50)
    parser.add_argument('--force_no_morph', type=int, default=0)

    # Augmentation
    parser.add_argument('--graphwithpca', type=bool, default=True)
    parser.add_argument('--uselabel', type=bool, default=False)
    parser.add_argument('--K_m0', type=int, default=7)
    parser.add_argument('--K_m1', type=int, default=7)

    # Cluster
    parser.add_argument('--cluster_using', type=str, default='gene_rec') # gene_rec
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--radius', type=int, default=50)
    parser.add_argument('--cluster_refinement', type=int, default=0)
    
    # Model
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--dim_input', type=int, default=3000)
    parser.add_argument('--dim_output', type=int, default=60)
    parser.add_argument('--alpha', type=float, default=0.0015)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--aug_rate_0', type=float, default=0.1)
    parser.add_argument('--aug_rate_1', type=float, default=0.1)
    parser.add_argument('--v_latent', type=float, default=0.05)
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_encoder_layer', type=int, default=1)
    parser.add_argument('--n_fusion_layer', type=int, default=1)
    parser.add_argument('--bn_type', type=str, default='bn')
    parser.add_argument('--self_loop', type=int, default=0)
    parser.add_argument('--down_sample_rate', type=float, default=1)
    parser.add_argument('--morph_trans_ratio', type=float, default=1)
    parser.add_argument('--aug_method', type=str, default="near_mix")
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--run_dir', type=str, default=os.getenv('WANDB_RUN_DIR'))
    parser.add_argument('--save_dir', type=str, default="result/")
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--var_plot', type=int, default=0)  # plot vairous cluster numbers
    parser.add_argument('--plot_louvain', type=int, default=0)
    parser.add_argument('--plot_leiden', type=int, default=0)
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
        os.makedirs(args.save_dir+'model/', exist_ok=True)
        with open(args.save_dir + 'setting.txt', 'w') as f:
            f.write(str(args))

    if not os.path.exists(args.R_HOME):
        raise EnvironmentError("R_HOME misconfigured. Run `Rscript -e 'R.home(component=\"home\")' ` and pass the output as R_HOME.")
    os.environ['R_HOME'] = args.R_HOME

    wandb_agent = wandb.init(
        project="MuST_main",
        entity="liliangyu",
        config=args.__dict__,
        name='MuST_'.join(sys.argv[1:]),
        mode=args.wandb,
        save_code=True,
        dir=args.run_dir,
        )
    
    visium = STData(name=args.dataset, crop_size=args.crop_size, bio_norm=False, sample=args.sample)   # Reset sample to get better results.

    adata = visium.adata
    adata.uns["name"] = args.dataset

    n_clusters = visium.get_annotation_class()
    if n_clusters is not None:
        warnings.warn("n_cluster rewritten due to known label")
    else:
        n_clusters = args.n_clusters

    # define model
    use_morph = None if args.force_no_morph else visium.get_morph()

    model = MUST(
        adata,
        use_morph,
        n_top_genes=args.n_top_genes,
        max_value=args.max_value,
        device=args.device,
        random_seed=args.random_seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        dim_input=args.dim_input,
        dim_output=args.dim_output,
        alpha=args.alpha,
        beta=args.beta,
        v_latent=args.v_latent,
        theta=args.theta,
        aug_rate_0=args.aug_rate_0,
        aug_rate_1=args.aug_rate_1,
        n_encoder_layer=args.n_encoder_layer,
        n_fusion_layer=args.n_fusion_layer,
        bn_type=args.bn_type, 
        self_loop=args.self_loop,
        morph_trans_ratio=args.morph_trans_ratio,
        graphwithpca=args.graphwithpca,
        uselabel=args.uselabel,
        K_m0=args.K_m0,
        K_m1=args.K_m1,
        aug_method=args.aug_method,
        unique_str=f"Crop{args.crop_size}",
        datatype=visium.platform,
        preprocessed=args.preprocessed,
        down_sample_rate=args.down_sample_rate,
        min_cells=args.min_cells,
    )


    # train model
    print("INPUT GENE SHAPE: ", model.adata.shape, "HVG Selected: ", model.adata.var['highly_variable'].sum())
    adata = model.train()

    wandb_logs = {}

    recon = adata.obsm['gene_rec']
    emb = adata.obsm['emb']
    cluster_emb = emb if args.cluster_using == 'emb' else recon
    if args.plot:
        if emb.shape[1] != 2:
            emb_2d = UMAP(n_components=2, random_state=args.random_seed).fit_transform(emb)
        else:
            emb_2d = emb

    if args.save_dir is not None:
        model.save(args.save_dir+'model/')
        # np.save(args.save_dir + 'recon.npy', adata.obsm['gene_rec'])
        np.save(args.save_dir + 'emb.npy', adata.obsm['emb'])
        # np.save(args.save_dir + 'trans_input.npy', adata.obsm['trans_input'])
        np.save(args.save_dir + 'emb_2d.npy', emb_2d)
        np.save(args.save_dir + 'hvg.npy', adata.var['highly_variable'].to_numpy())
        # np.save(args.save_dir + 'trans_emb', model.trans_emb)
        # if model.morph_emb is not None:
        #     np.save(args.save_dir + 'morph_emb', model.morph_emb)

    # kmeans_pred = cluster(cluster_emb, method="kmeans", n_clusters=n_clusters)
    if args.plot_louvain:
        louvain_pred = targeted_cluster(cluster_emb, method="louvain", target_n_clusters=n_clusters)
    if args.plot_leiden:
        leiden_pred = targeted_cluster(cluster_emb, method="leiden", target_n_clusters=n_clusters)
    # mclust_pred = cluster(cluster_emb, method="mclust", n_clusters=n_clusters, pca_dim=20)
    mclust_pred = stable_cluster(cluster_emb, method="mclust", n_clusters=n_clusters, pca_dim=20)
    if args.cluster_refinement: # Hexogonal Refinement
        print("REFINE!")
        # kmeans_pred = refine_label(kmeans_pred, corrds=visium.get_coords(), radius=args.radius)
        if args.plot_louvain:
            louvain_pred = refine_label(louvain_pred, corrds=visium.get_coords(), radius=args.radius)
        if args.plot_leiden:
            leiden_pred = refine_label(leiden_pred, corrds=visium.get_coords(), radius=args.radius)
        mclust_pred = refine_label(mclust_pred, corrds=visium.get_coords(), radius=args.radius)
    if visium.get_label() is not None:
        true = visium.get_label()
        # kmeans_pred = cluster_map(true, kmeans_pred, wildcard=999)
        if args.plot_louvain:
            louvain_pred = cluster_map(true, louvain_pred, wildcard=999)
        if args.plot_leiden:
            leiden_pred = cluster_map(true, leiden_pred, wildcard=999)
        mclust_pred = cluster_map(true, mclust_pred, wildcard=999)

    if args.save_dir is not None:
        np.save(args.save_dir + 'pred_main.npy', mclust_pred)
        if args.plot_louvain:
            np.save(args.save_dir + 'pred_louvain.npy', louvain_pred)
        if args.plot_leiden:
            np.save(args.save_dir + 'pred_leiden.npy', leiden_pred)

    ecb_e_trans = ecb.Eval(input=model.input_trans, latent=emb, label=mclust_pred, k=10)
    trans_mrre_zx, trans_mrre_xz = ecb_e_trans.E_mrre()
    trans_mrre = np.mean([trans_mrre_xz, trans_mrre_zx])
    wandb_logs.update({
        f"metrics/MRRE_trans_{10}": trans_mrre,
        f"metrics/MRRE_trans_xz_{10}": trans_mrre_xz,
        f"metrics/MRRE_trans_zx_{10}": trans_mrre_zx,
        f"metrics/mclust_sc": silhouette_score(emb, mclust_pred),
        f"metrics/mclust_db": davies_bouldin_score(emb, mclust_pred),
    })
    if visium.get_morph() is not None:
        ecb_e_morph = ecb.Eval(input=visium.get_morph(), latent=emb, label=mclust_pred, k=10)

        morph_mrre_zx, morph_mrre_xz = ecb_e_morph.E_mrre()
        morph_mrre = np.mean([morph_mrre_xz, morph_mrre_zx])
        mrre_xz = morph_mrre_xz + trans_mrre_xz
        mrre_zx = morph_mrre_zx + trans_mrre_zx

        wandb_logs.update({
            f"metrics/MRRE_{10}": morph_mrre + trans_mrre,
            f"metrics/MRRE_morph_{10}": morph_mrre,
            f"metrics/MRRE_morph_xz_{10}": morph_mrre_xz,
            f"metrics/MRRE_xz_{10}": mrre_xz,
            f"metrics/MRRE_morph_zx_{10}": morph_mrre_zx,
            f"metrics/MRRE_zx_{10}": mrre_zx,
        })

    if args.plot:
        # fig_spatial_plain_kmeans = visium.px_plot_spatial(kmeans_pred, background_image=False)
        if args.save_dir is None:
            fig_spatial_plain_mclust = visium.px_plot_spatial(mclust_pred, background_image=False)
            fig_emb_mclust = visium.px_plot_embedding(emb_2d, mclust_pred)
            if args.plot_louvain:
                fig_spatial_plain_louvain = visium.px_plot_spatial(louvain_pred, background_image=False)
                fig_emb_louvain = visium.px_plot_embedding(emb_2d, louvain_pred)
            if args.plot_leiden:
                fig_spatial_plain_leiden = visium.px_plot_spatial(leiden_pred, background_image=False)
                fig_emb_leiden = visium.px_plot_embedding(emb_2d, leiden_pred)
        else:
            fig_spatial_plain_mclust = visium.px_plot_spatial(mclust_pred, background_image=visium.platform == '10x', save_path=args.save_dir+'mclust.png')
            fig_emb_mclust = visium.px_plot_embedding(emb_2d, mclust_pred, save_path=args.save_dir+'mclust_emb.png')
            if args.plot_louvain:
                fig_spatial_plain_louvain = visium.px_plot_spatial(louvain_pred, background_image=visium.platform == '10x', save_path=args.save_dir+'louvain.png')
                fig_emb_louvain = visium.px_plot_embedding(emb_2d, louvain_pred, save_path=args.save_dir+'louvian_emb.png')
            if args.plot_leiden:
                fig_spatial_plain_leiden = visium.px_plot_spatial(leiden_pred, background_image=visium.platform == '10x', save_path=args.save_dir+'leiden.png')
                fig_emb_leiden = visium.px_plot_embedding(emb_2d, leiden_pred,  save_path=args.save_dir+'leiden_emb.png')
        wandb_logs.update({
            # "figs/plain Spatial Image - KMeans": fig_spatial_plain_kmeans,
            "figs/plain Spatial Image - mclust": fig_spatial_plain_mclust,
            'figs/UMAP mclust': fig_emb_mclust,
        })
        if args.plot_louvain:
            wandb_logs.update({
                "figs/plain Spatial Image - Louvain": fig_spatial_plain_louvain,
                'figs/UMAP louvain': fig_emb_louvain,
            })
        if args.plot_leiden:
            wandb_logs.update({
                "figs/plain Spatial Image - Leiden": fig_spatial_plain_leiden,
                'figs/UMAP leiden': fig_emb_leiden,
            })
        if args.var_plot:
            for var_cluster_num in [ 20 ]:
                var_mclust_pred = stable_cluster(emb, method="mclust", n_clusters=var_cluster_num, pca_dim=20)
                if args.save_dir is not None:
                    np.save(args.save_dir + f'pred_{var_cluster_num}.npy', var_mclust_pred)
                    fig_spatial_var_mclust = visium.px_plot_spatial(var_mclust_pred, background_image=True, save_path=args.save_dir+f'mclust_{var_cluster_num}.png')
                    fig_embedding_var_mclust = visium.px_plot_embedding(emb_2d, var_mclust_pred, save_path=args.save_dir+f'mclust_{var_cluster_num}_emb.png')
                else:
                    fig_spatial_var_mclust = visium.px_plot_spatial(var_mclust_pred, background_image=False)
                    fig_embedding_var_mclust = visium.px_plot_embedding(emb_2d, var_mclust_pred)
                wandb_logs.update({
                    f"figs_variety/spatial_mclust_class{var_cluster_num}": fig_spatial_var_mclust,
                    f"figs_variety/UMAP_mclust_class{var_cluster_num}": fig_embedding_var_mclust,
                })

    if visium.get_label() is not None:
        mask = true != 999
        mclust_error = make_error_label(true, mclust_pred, wildcard=999)

        wandb_logs.update({
            # f"metrics/k_means_acc": aligned_accuracy_score(true[mask], kmeans_pred[mask], wildcard=999),
            # f"metrics/k_means_ari": adjusted_rand_score(true[mask], kmeans_pred[mask]),
            # f"metrics/k_means_nmi": normalized_mutual_info_score(true[mask], kmeans_pred[mask]),
            # f"metrics/k_means_ami": adjusted_mutual_info_score(true[mask], kmeans_pred[mask]),
            # f"metrics/louvain_acc": aligned_accuracy_score(true[mask], louvain_pred[mask], wildcard=999),
            # f"metrics/louvain_ari": adjusted_rand_score(true[mask], louvain_pred[mask]),
            # f"metrics/louvain_nmi": normalized_mutual_info_score(true[mask], louvain_pred[mask]),
            # f"metrics/louvain_ami": adjusted_mutual_info_score(true[mask], louvain_pred[mask]),
            # f"metrics/mclust_acc": aligned_accuracy_score(true[mask], mclust_pred[mask], wildcard=999),
            f"metrics/mclust_ari": adjusted_rand_score(true[mask], mclust_pred[mask]),
            # f"metrics/mclust_nmi": normalized_mutual_info_score(true[mask], mclust_pred[mask]),
            # f"metrics/mclust_ami": adjusted_mutual_info_score(true[mask], mclust_pred[mask]),
        })
        if args.plot:
            # fig_spatial_plain_mclust_error = visium.px_plot_spatial(mclust_error, background_image=False)
            # fig_spatial_plain_true = visium.px_plot_spatial(true, background_image=False)
            if args.save_dir is not None:
                fig_embedding_true = visium.px_plot_embedding(emb_2d, true, save_path=args.save_dir+'true_emb.png')
            else:
                fig_embedding_true = visium.px_plot_embedding(emb_2d, true)
            wandb_logs.update({
                # f"figs/plain Spatial Image - True": fig_spatial_plain_true,
                # f"figs/plain Spatial Image - mclust Error": fig_spatial_plain_mclust_error,
                f'figs/UMAP true': fig_embedding_true,
            })

    save_file_path = f'{wandb.run.dir[:-5]}flag.txt'
    wandb.log(wandb_logs)
    wandb.finish()

    with open(save_file_path, 'w') as f:
        f.write('finish run all')