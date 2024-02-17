import os
import urllib
import warnings
import numpy as np
import scanpy as sc
from tqdm import tqdm, trange

import PIL
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class STData():
    def __init__(self, name="V1_Adult_Mouse_Brain", sample='barcode', crop_size=None, use_quality="hires", use_adata=None, bio_norm=True, force_no_morph=False,
                 gene_dim=1000, gene_decompn_method="HVG", seed=0, datapath="data/", tmppath="save_processed_data/", resize_as=None, true_wildcard=999):
        # Globals
        PIL.Image.MAX_IMAGE_PIXELS = 30000 * 30000
        self.seed = seed
        self.name = name
        self.force_no_morph = force_no_morph
        self.use_quality = use_quality
        self.true_wildcard = true_wildcard
        self.platform = '10x'

        self.datapath = datapath
        self.tmppath = tmppath
        os.makedirs(self.datapath, exist_ok=True)
        os.makedirs(self.tmppath, exist_ok=True)

        # pack dataset from existing adata
        if use_adata is None:
            if name in ["190921_19", "191007_07", "200127_15", "190926_06", "190926_02", "191204_01",
                        "190921_21", "190926_01", "200306_03", "190926_03", "200115_08"]:
                self.platform = 'slide'
                self.adata = sc.read(f"{self.datapath}{name}/data.h5ad")
            elif name in ['MOB', 'MOB_3000', 'ME95', 'ME145', 'ME145_3000']:
                self.platform = 'stereo'
                self.adata = sc.read(f"{self.datapath}{name}/data.h5ad")
            else:
                try:
                    self.adata = sc.read_visium(self.datapath + name)
                except FileNotFoundError:
                    try:
                        self.adata = sc.datasets.visium_sge(self.name,
                                                            include_hires_tiff=True if use_quality == "fulres" else False)
                    except urllib.error.HTTPError:
                        raise NotImplementedError("Non-Implmented Dataset! You can put your own `self.adata` here")
            self.adata.var_names_make_unique()
        else:
            self.adata = use_adata
        
        # ditch non-mark spots if available
        if true_wildcard is not None and self.get_label() is not None:
            self.true = self.get_label()
            self.mask = self.true != true_wildcard
            if len(self.adata) != sum(self.mask):
                self.adata = self.adata[self.mask, :]
            self.true = self.true[self.mask]

        # sampling
        np.random.seed(self.seed)
        if isinstance(sample, int) and sample < len(self.adata):
            if sample == 15000:
                print("default sampling!")
            sc.pp.filter_cells(self.adata, min_genes=0)
            n_genes = self.adata.obs['n_genes'].to_numpy()
            idxs = np.argsort(n_genes)
            self.selected_idx = idxs[-sample:]
            self.adata = self.adata[self.selected_idx]
        elif isinstance(sample, str) and sample == 'barcode':
            try:
                selected = np.load(f"{self.datapath}{name}/selected_spots.npy")
                self.adata = self.adata[selected]
            except FileNotFoundError:
                warnings.warn('Try to use selected spots, but not found!')
        elif isinstance(sample, str) and sample.startswith('uni'):
            idxs = np.arange(len(self.adata))
            np.random.shuffle(idxs)
            self.selected_idx = idxs[:eval(sample[3:])]
            self.adata = self.adata[self.selected_idx]

        # Gene-related
        self.gene_dim = gene_dim
        self.gene_decompn_method = gene_decompn_method

        if bio_norm:
            self.default_norm()

        # Image-related
        self.coords = np.asarray(self.adata.obsm["spatial"], dtype=np.float32)
        self.x_lim = (int(min(self.coords[:, 0])), int(max(self.coords[:, 0])))
        self.y_lim = (int(max(self.coords[:, 1])), int(min(self.coords[:, 1])))
        if self.platform == '10x':
            self.resize_as = resize_as
            self.crop_size = 224 if crop_size is None else crop_size

            if self.use_quality == "fulres":
                path = f"{self.datapath}/{self.name}/image.tif"
                self.image = np.asarray(Image.open(path), dtype=np.float32)
                self.image /= 255
            else:
                self.image = self.adata.uns["spatial"][name]["images"][use_quality]
            self.x_lim = (0, self.image.shape[1])
            self.y_lim = (self.image.shape[0], 0)

            if use_quality != "fulres":
                self.coords *= self.adata.uns["spatial"][name]["scalefactors"][f"tissue_{use_quality}_scalef"]

    def default_norm(self):
        sc.pp.filter_genes(self.adata, min_cells=1)
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)
        if self.gene_dim != -1 and self.gene_decompn_method == "HVG":
            sc.pp.highly_variable_genes(self.adata, n_top_genes=self.gene_dim)
            self.adata = self.adata[:, self.adata.var.highly_variable]

    def get_morph(self, use_saved_data=True):
        if self.platform in ['slide', 'stereo'] or self.force_no_morph:
            return None

        path = self.tmppath + f"morph_{self.name}_{self.use_quality}_{self.crop_size}_{self.resize_as}_{self.seed}_{self.true_wildcard}_{len(self.adata)}_ver230509.npy"
        if os.path.exists(path) and use_saved_data:
            return np.load(path, allow_pickle=True)

        import torch
        from torchvision.models import resnet50
        torch.manual_seed(self.seed)

        model = resnet50(pretrained=True)
        model.fc = torch.nn.Identity()
        model.cuda()
        model.eval()

        image = self.image
        morph_feature = []
        if len(self.image.shape) == 2:
            image = self.image[:, :, np.newaxis]
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                          bias=False).cuda()
        for coord in tqdm(self.coords, desc="Extracting Morphological Feature"):
            left = max(int(coord[0] - self.crop_size / 2), 0)
            up = max(int(coord[1] - self.crop_size / 2), 0)
            # Approximate Upperbound
            right = min(left + self.crop_size, self.y_lim[0])
            down = min(up + self.crop_size, self.x_lim[1])
            pic = image[up:down, left:right]
            if self.resize_as is not None:
                from PIL import Image
                pic = Image.fromarray(np.uint8(pic * 255))
                pic = pic.resize((self.resize_as, self.resize_as))
                pic = np.asarray(pic).astype(np.float32) / 255
            pic = np.transpose(pic, (2, 0, 1))
            feature = model(torch.tensor(np.expand_dims(pic, axis=0)).cuda())
            morph_feature.append(feature.detach().cpu().numpy().squeeze())

        morph_feature = PCA(n_components=500, random_state=self.seed).fit_transform(morph_feature)

        if use_saved_data:
            np.save(path, morph_feature.astype(np.float32))

        return morph_feature.astype(np.float32)

    def get_spot_index(self):
        return self.adata.obs[["array_col", "array_row"]].to_numpy()

    def get_trans(self):
        try:
            trans = np.array(self.adata.X.todense(), dtype=np.float32)
        except AttributeError:  # raised by non-sparse matrix
            trans = np.asarray(self.adata.X, dtype=np.float32)

        if self.gene_decompn_method == "PCA":
            trans = PCA(n_components=self.gene_dim, random_state=self.seed).fit_transform(trans)

        return trans

    def get_image(self):
        return self.image

    def get_annotation_class(self, suffix="biofeature"):
        import pandas as pd

        data = pd.read_csv("annotation/dataset_setting.csv", index_col="dataset")
        data.index = data.index.astype(str)
        try:
            data = data[data["suffix"] == suffix].loc[self.name]
        except KeyError:
            return None

        return int(data["class_num"])

    def get_label(self, suffix="biofeature", use_saved=True, refine=None):
        if hasattr(self, 'true'):
            return self.true

        path = f"annotation/{self.name}_{suffix}"
        if use_saved:
            try:
                return np.load(path + ".npy")
            except FileNotFoundError:
                warnings.warn("Try to use true label, but not found!")

        from sklearn.cluster import KMeans

        # make true label from annotation
        try:
            im = np.array(Image.open(path + ".png"))
        except:
            return None

        n_clusters = self.get_annotation_class(suffix=suffix)
        coords = self.adata.obsm["spatial"].astype(np.float32)
        coords *= self.adata.uns["spatial"][self.name]["scalefactors"]["tissue_hires_scalef"]
        labels = []
        for coord in coords:
            val = im[int(coord[1]), int(coord[0])]
            val = val[0] + val[1] * 255 + val[2] * 255 * 255 + val[3] * 255 * 255 * 255
            labels.append([val])
        labels = np.array(labels)
        labels = KMeans(n_clusters=n_clusters, random_state=self.seed).fit_predict(labels)

        if refine == "hexagon":
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(radius=30).fit(self.get_coords())
            idxs = nbrs.radius_neighbors(self.get_coords(), return_distance=False)

            for i, vec in enumerate(idxs):
                vec_label = np.vectorize(labels.__getitem__)(vec)
                if np.sum(vec_label == labels[i]) <= 1:
                    labels[i] = np.argmax(np.bincount(vec_label))

        np.save(path + ".npy", labels)

        return np.array(labels)

    def get_coords(self):
        return np.array(self.coords)

    def filter_sect(self, x1, x2, y1, y2):
        xs = self.coords[:, 0]
        ys = self.coords[:, 1]
        m_x = (x1 < xs) & (xs < x2)
        m_y = (y1 < ys) & (ys < y2)
        mask = m_x & m_y
        self.coords = self.coords[mask]
        self.adata = self.adata[mask, :]

        return mask

    def filter_region(self, label, regions):
        mask = np.array([False] * len(self.adata))
        for r in regions:
            mask |= (label == r)
        self.coords = self.coords[mask]
        self.adata = self.adata[mask, :]

        return mask
    
    def find_HVG(self, label:np.ndarray, num_gene=50, flavor='seurat_v3'):
        res = {}
        for target_cluster in np.unique(label):
            uni_cluster_adata = self.adata[label == target_cluster, :]
            sc.pp.highly_variable_genes(uni_cluster_adata, n_top_genes=num_gene, inplace=True, flavor=flavor)
            res[target_cluster] = np.asarray(uni_cluster_adata[:, uni_cluster_adata.var.highly_variable].var_names).astype(str)

        return res
    
    def get_related_gene(self, label, c, ret_val=False, norm=False):
        gene = self.get_trans()
        direct = np.ones(len(self.adata))
        direct[label != c] *= -1
        val = np.sum(gene.T * direct, axis=1)
        if norm:
            val /= np.sum(gene, axis=0)
        idxs = np.flip(np.argsort(val))
        names = np.asarray(self.adata.var_names).astype(str)
        genes = names[idxs]

        if ret_val:
            return genes, val[idxs]
        else:
            return genes

    def get_plot_bg_image(self):
        bg_path = f'background_image/{self.name}.png'
        if os.path.exists(bg_path):
            return np.asarray(Image.open(bg_path), dtype=np.float32) / 255
        else:
            return self.image

    def px_plot_spatial(self, label: np.ndarray, background_image=False, save_path=None, base=None, dpi=400, title=None, opacity=None):
        import plotly.express as px
        import plotly.graph_objects as go

        if self.platform != '10x' and background_image:
            warnings.warn("no bg image for non-10x platform")
            background_image = False

        if opacity is None:
            opacity = .3 if background_image else 1
        title = f"{len(np.unique(label))} classes" if title is None else title

        # Save Figure
        if save_path is not None:
            import matplotlib.pyplot as plt
            s = 4 if self.platform == '10x' else 1
            if self.platform == '10x':
                figsize = (5, 5)
            elif self.platform == 'stereo':
                ratio = np.abs(self.y_lim[0] - self.y_lim[1]) / np.abs(self.x_lim[0] - self.x_lim[1])
                if base is None and self.name == 'ME145':
                    base = 10
                elif self.name == 'ME95':
                    base = 3
                elif self.name == 'MOB':
                    base = 6
                figsize = (base, base * ratio)
            else:
                figsize = (7, 7)

            fig_plt, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.set_xlim([self.x_lim[0], self.x_lim[1]])
            ax.set_ylim([self.y_lim[0], self.y_lim[1]])
            if background_image:
                ax.imshow(self.get_plot_bg_image())
            ax.scatter(x=self.coords[:, 0], y=self.coords[:, 1], c=self._map_color(label), s=s, alpha=opacity)
            plt.axis('off')

            fig_plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        if background_image:
            if self.use_quality == "fulres":
                warnings.warn("Plotting with fulres may lead to a figure taking huge memory.")
            fig_plotly = px.imshow(self.get_plot_bg_image())
        else:
            fig_plotly = go.Figure()

        for i in np.unique(label):
            mask = label == i
            idxs = np.arange(len(label))[mask]
            fig_plotly.add_trace(go.Scatter(
                mode='markers',
                x=self.coords[mask, 0],
                y=self.coords[mask, 1],
                opacity=opacity,
                marker=dict(
                    color=self._map_color(label[mask]),
                    size= 10 if self.platform == '10x' else 2
                ),
                hovertemplate="%{text}",
                text=[f"Spot: {self.adata.obs.index[i]}<br>"
                      f"x: {self.coords[i, 0]:.2f}<br>"
                      f"y: {self.coords[i, 1]:.2f}<br>"
                      f"index: {i}"
                      for i in idxs],
                showlegend=True,
                name=str(i)
            ))
        fig_plotly.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig_plotly.update_layout(
            title=title,
            xaxis_range=[self.x_lim[0], self.x_lim[1]],
            yaxis_range=[self.y_lim[0], self.y_lim[1]]
        )

        return fig_plotly


    def px_plot_spatial_gene(self, gene, background_image=False, save_path=None, style='sequential', title=None, dpi=400):
        import plotly.express as px
        import plotly.graph_objects as go

        opacity = .5 if background_image else 1
        if isinstance(gene, str):
            try:
                data = np.asarray(self.adata[:, gene].X.todense()).ravel().astype(np.float32)
            except AttributeError:
                data = np.asarray(self.adata[:, gene].X).ravel().astype(np.float32)
            data /= data.max()
        else:
            data = gene
            gene = 'user_defined'

        title = gene if title is None else title

        # Save Figure
        if save_path is not None:
            import matplotlib.pyplot as plt
            s = 4 if self.platform == '10x' else 1
            if self.platform == '10x':
                figsize = (5, 5)
            elif self.platform == 'stereo':
                ratio = np.abs(self.y_lim[0] - self.y_lim[1]) / np.abs(self.x_lim[0] - self.x_lim[1])
                if self.name == 'ME145':
                    base = 8
                elif self.name == 'ME95':
                    base = 3
                elif self.name == 'MOB':
                    base = 6
                figsize = (base, base * ratio)
            else:
                figsize = (7, 7)

            fig_plt, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.set_xlim([self.x_lim[0], self.x_lim[1]])
            ax.set_ylim([self.y_lim[0], self.y_lim[1]])
            if background_image:
                ax.imshow(self.get_plot_bg_image())
            ax.scatter(x=self.coords[:, 0], y=self.coords[:, 1], c=data, s=s, alpha=opacity, cmap='RdPu' if style=='sequential' else 'seismic')   # seismic
            plt.axis('off')

            fig_plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        if background_image:
            if self.use_quality == "fulres":
                warnings.warn("Plotting with fulres may lead to a figure taking huge memory.")
            fig_plotly = px.imshow(self.get_plot_bg_image())
        else:
            fig_plotly = go.Figure()

        idxs = np.arange(len(data))
        fig_plotly.add_trace(go.Scatter(
            mode='markers',
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            opacity=opacity,
            marker=dict(
                color=data,
                colorscale="RdPu" if style=='sequential' else 'RdBu',  # sequential RdPu diverging RdBu
            ),
            hovertemplate="%{text}",
            text=[f"Spot: {self.adata.obs.index[i]}<br>"
                  f"Expr_ratio: {data[i]:.2f}<br>"
                    f"x: {self.coords[i, 0]:.2f}<br>"
                    f"y: {self.coords[i, 1]:.2f}<br>"
                    f"index: {i}"
                    for i in idxs],
            showlegend=True,
            name=gene,
        ))
        fig_plotly.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig_plotly.update_layout(
            title=title,
            xaxis_range=[0, self.x_lim],
            yaxis_range=[self.y_lim, 0]
        )

        return fig_plotly
    

    def px_plot_spatial_massive_gene(self, genes, col_num=7, figsize=400, markersize=2, save_path=None, title=None, background_image=False, verbose=False):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        opacity = .5 if background_image else 1
        
        try:
            data = self.adata[:, genes].X.todense()
        except AttributeError:
            data = self.adata[:, genes].X
        data = np.asarray(data, dtype=np.float32)
        data /= data.max()

        row_num = np.ceil(len(genes)/col_num).astype(int).item()
        fig_plotly = make_subplots(rows=row_num, cols=col_num,
                                    subplot_titles=genes)

        if verbose:
            tr = trange(data.shape[1])
        else:
            tr = range(data.shape[1])
        for i in tr:
            fig_plotly.add_trace(go.Scatter(
                mode='markers',
                x=self.coords[:, 0],
                y=self.coords[:, 1],
                opacity=opacity,
                marker=dict(
                    color=data[:, i],
                    colorscale="Magenta",
                ),
                hovertemplate="%{text}",
                text=[f"Spot: {spot}<br>x: {coord[0]:.2f}<br>y: {coord[1]:.2f}"
                        for spot, coord in zip(np.array(self.adata.obs.index), self.coords)],
                showlegend=True,
                name=genes[i],
            ), row=i // col_num + 1, col=i % col_num + 1)
        fig_plotly.update_layout({
            "plot_bgcolor": "#ffffff",
            "paper_bgcolor": "#ffffff",
        })
        fig_plotly.update_layout(
            width=figsize * col_num,
            height=figsize * row_num,
            title_text=title,
        )
        fig_plotly.update_xaxes(visible=False)
        fig_plotly.update_yaxes(visible=False)

        if save_path is not None:
            if save_path.endswith('html'):
                fig_plotly.write_html(save_path)
            else:
                fig_plotly.write_image(save_path, scale=4)

        return fig_plotly


    def px_plot_graph_structure(self, edge_list: np.ndarray, background_image=False, title=None):
        import plotly.express as px
        import plotly.graph_objects as go

        lines = self.coords[edge_list]

        if background_image:
            if self.use_quality == "fulres":
                warnings.warn("Plotting with fulres may lead to a huge figure while taking huge memory.")
            fig_plotly = px.imshow(self.get_plot_bg_image())
        else:
            fig_plotly = go.Figure()
        
        x = lines[:, :, 0].reshape(-1)
        y = lines[:, :, 1].reshape(-1)

        x = np.insert(x, np.arange(2, len(x), 2), np.nan)
        y = np.insert(y, np.arange(2, len(y), 2), np.nan)

        fig_plotly.add_trace(go.Scatter(
            mode='lines',
            x=x,
            y=y,
            line=dict(
                # color="black",
            )
        ))
        fig_plotly.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig_plotly.update_layout(
            title=title,
            xaxis_range=[self.x_lim[0], self.x_lim[1]],
            yaxis_range=[self.y_lim[0], self.y_lim[1]]
        )

        return fig_plotly


    def px_plot_embedding(self, latent, label, method="UMAP", save_path=None, title=None):
        import plotly.graph_objects as go

        title = f"{len(np.unique(label))} classes" if title is None else title

        if latent.shape[1] == 2:
            embedding = latent
        elif method == "UMAP":
            from umap import UMAP
            embedding = UMAP(n_components=2, random_state=self.seed).fit_transform(latent)
        elif method == "TSNE":
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2, random_state=self.seed).fit_transform(latent)
        else:
            raise ValueError(f"{method} not valid!")

        # Save Figure
        if save_path is not None:
            import matplotlib.pyplot as plt
            figsize = (7, 7) if self.platform == '10x' else (10, 10)

            fig_plt, ax = plt.subplots(figsize=figsize, dpi=400)
            ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=self._map_color(label), s=1)
            plt.axis('off')

            fig_plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        fig_plotly = go.Figure()
        for i in np.unique(label):
            mask = label == i
            fig_plotly.add_trace(go.Scatter(
                mode='markers',
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                marker=dict(
                    color=self._map_color(label[mask]),
                    size=3
                ),
                hovertemplate="%{text}",
                text=[f"Spot: {spot}<br>x: {coord[0]:.2f}<br>y: {coord[1]:.2f}"
                      for spot, coord in zip(np.array(self.adata.obs.index)[mask], self.coords[mask])],
                showlegend=True,
                name=str(i)
            ))
        fig_plotly.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig_plotly.update_layout(
            title=title,
        )

        return fig_plotly


    def px_plot_embedding_gene(self, latent, gene, method="UMAP", save_path=None, title=None):
        import plotly.graph_objects as go

        title = gene if title is None else title
        data = np.asarray(self.adata[:, gene].X.todense()).ravel().astype(np.float32)
        data /= data.max()

        if latent.shape[1] == 2:
            embedding = latent
        elif method == "UMAP":
            from umap import UMAP
            embedding = UMAP(n_components=2, random_state=self.seed).fit_transform(latent)
        elif method == "TSNE":
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2, random_state=self.seed).fit_transform(latent)
        else:
            raise ValueError(f"{method} not valid!")

        # Save Figure
        if save_path is not None:
            import matplotlib.pyplot as plt

            fig_plt, ax = plt.subplots(figsize=(5, 5), dpi=400)
            ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=data, s=2)
            plt.axis('off')

            fig_plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(
            mode='markers',
            x=embedding[:, 0],
            y=embedding[:, 1],
            marker=dict(
                color=data,
                colorscale="PuRd",
            ),
            hovertemplate="%{text}",
            text=[f"Spot: {spot}<br>x: {coord[0]:.2f}<br>y: {coord[1]:.2f}"
                    for spot, coord in zip(np.array(self.adata.obs.index), self.coords)],
            showlegend=True,
            name=gene
        ))
        fig_plotly.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig_plotly.update_layout(
            title=title,
        )

        return fig_plotly
    
    def px_plot_embedding_massive_gene(self, latent, genes, col_num=7, figsize=400, markersize=2, method="TSNE", save_path=None, title=None):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        data = np.asarray(self.adata[:, genes].X.todense()).astype(np.float32)
        data /= data.max()

        if latent.shape[1] == 2:
            embedding = latent
        elif method == "UMAP":
            from umap import UMAP
            embedding = UMAP(n_components=2, random_state=self.seed).fit_transform(latent)
        elif method == "TSNE":
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2, random_state=self.seed).fit_transform(latent)
        else:
            raise ValueError(f"{method} not valid!")

        row_num = np.ceil(len(genes)/col_num).astype(int).item()
        fig_plotly = make_subplots(rows=row_num, cols=col_num,
                                    subplot_titles=genes)
        for i in range(data.shape[1]):
            fig_plotly.add_trace(go.Scatter(
                mode='markers',
                x=embedding[:, 0],
                y=embedding[:, 1],
                marker=dict(
                    color=data[:, i],
                    colorscale="Magenta",
                    size=markersize,
                ),
                hovertemplate="%{text}",
                text=[f"Spot: {spot}<br>x: {coord[0]:.2f}<br>y: {coord[1]:.2f}"
                        for spot, coord in zip(np.array(self.adata.obs.index), self.coords)],
                showlegend=True,
                name=genes[i]
            ), row=i // col_num + 1, col=i % col_num + 1)
        fig_plotly.update_layout({
            "plot_bgcolor": "#ffffff",
            "paper_bgcolor": "#ffffff",
        })
        fig_plotly.update_layout(
            width=figsize * col_num,
            height=figsize * row_num,
            title_text=title,
        )
        fig_plotly.update_xaxes(visible=False)
        fig_plotly.update_yaxes(visible=False)

        if save_path is not None:
            if save_path.endswith('html'):
                fig_plotly.write_html(save_path)
            else:
                fig_plotly.write_image(save_path, scale=4)

        return fig_plotly


    def _map_color(self, label: np.ndarray, style='p_d24') -> list:
        if style == 'bright':
            color_scheme = ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4',
                            '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616',
                            '#479B55', '#EEA6FB', '#DC587D', '#D626FF',
                            '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5',
                            '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00',
                            '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']
        elif style == 'dark':
            color_scheme = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2",
                            "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
                            "#9c755f", "#bab0ab", "#1b9e77", "#d95f02",
                            "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
                            "#a6761d", "#666666"]
        elif style == 'p_d24':
            color_scheme = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D',
                            '#DA16FF', '#222A2A', '#B68100', '#750D86',
                            '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
                            '#FC0080', '#B2828D', '#6C7C32', '#778AAE',
                            '#862A16', '#A777F1', '#620042', '#1616A7', 
                            '#DA60CA', '#6C4516', '#0D2A63', '#AF0038',
                            '#D3D3D3']

        return [color_scheme[i % len(color_scheme)] for i in label]

    def __len__(self):
        return len(self.adata)
