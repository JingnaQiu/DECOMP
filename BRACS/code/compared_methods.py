from dataloader import make_RoI_loader
from utils import *
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F


class HACTModel_graph_embeddings(HACTModel):
    def calculate_graph_embedding(self,
                                  cell_graph: Union[dgl.DGLGraph, dgl.batch],
                                  tissue_graph: Union[dgl.DGLGraph, dgl.batch],
                                  assignment_matrix: torch.Tensor
                                  ) -> torch.Tensor:
        # 1. GNN layers over the low level graph
        ll_feats = cell_graph.ndata[GNN_NODE_FEAT_IN]  # (nr_cells, 512)
        ll_h = self.cell_graph_gnn(cell_graph, ll_feats, with_readout=False)  # (nr_cells, 64)

        # 2. Sum the low level features according to assignment matrix
        ll_h_concat = self._compute_assigned_feats(
            cell_graph, ll_h, assignment_matrix)  # (nr_sp, 64)

        tissue_graph.ndata[GNN_NODE_FEAT_IN] = torch.cat(
            (ll_h_concat, tissue_graph.ndata[GNN_NODE_FEAT_IN]), dim=1)  # (nr_sp, 64+514=578)

        # 3. GNN layers over the high level graph
        hl_feats = tissue_graph.ndata[GNN_NODE_FEAT_IN]
        graph_embeddings = self.superpx_gnn(tissue_graph, hl_feats)  # (bs, 64)

        return graph_embeddings
    
    def calculate_logits(self, graph_embeddings):
        logits = self.pred_layer(graph_embeddings)  # (bs, n_class)
        return logits

def get_model_dataloader(cfg, checkpoint, candidate_RoI_metas, set_name):
    model = HACTModel_graph_embeddings(
        cg_gnn_params=cfg.model_config['cg_gnn_params'],
        tg_gnn_params=cfg.model_config['tg_gnn_params'],
        classification_params=cfg.model_config['classification_params'],
        cg_node_dim=NODE_DIM,
        tg_node_dim=NODE_DIM,
        num_classes=cfg.n_classes,
    )

    model = model.to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(checkpoint))

    dataloader = make_RoI_loader(
        RoIs=[meta["RoI"] for meta in candidate_RoI_metas],
        cg_path=os.path.join(cfg.cg_path, set_name),
        tg_path=os.path.join(cfg.tg_path, set_name),
        assign_mat_path=os.path.join(cfg.assign_mat_path, set_name),
        batch_size=cfg.batch_size,
        shuffle=False,
        load_in_ram=False,
        task_3_classes=True if cfg.task == "3-classes" else False,
    )
    return model, dataloader


def calculate_region_feature(cfg, checkpoint, candidate_RoI_metas, set_name):
    model, dataloader = get_model_dataloader(cfg, checkpoint, candidate_RoI_metas, set_name)

    candidate_RoI_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch[:-1]
            feat = model.calculate_graph_embedding(*data)  # (bs, 128)
            feat /= torch.linalg.vector_norm(feat, dim=1, keepdim=True)
            candidate_RoI_features.append(feat)

    candidate_RoI_features = torch.vstack(candidate_RoI_features).cpu().numpy()
    return candidate_RoI_features


def calculate_region_gradient_embedding(cfg, checkpoint, candidate_RoI_metas, set_name):
    model, dataloader = get_model_dataloader(cfg, checkpoint, candidate_RoI_metas, set_name)

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch[:-1]
            feat = model.calculate_graph_embedding(*data)  # (bs, embDim)
            logits = model.calculate_logits(feat)  # (bs, n_class)
            
            feat /= torch.linalg.vector_norm(feat, dim=1, keepdim=True)  # (bs, embDim)
            probs = F.softmax(logits, dim=1)  # (bs, n_class)
            preds = probs.argmax(dim=1)  # (bs,)

            one_hot = F.one_hot(preds, num_classes=logits.shape[1]).float()  # (bs, n_class)
            delta = one_hot - probs  # (bs, n_class)

            g = torch.einsum('bi,bj->bij', feat, delta)  # (bs, embDim, num_cls)
            embeddings.append(g.reshape(feat.shape[0], -1))    # (bs, embDim*num_cls)

    embeddings = torch.vstack(embeddings).cpu().numpy()
    return embeddings


# https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
class kCenterGreedy():
    def __init__(self, X, metric='euclidean'):
        self.features = X
        self.name = 'kcenter'
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.features.shape[0]

    def update_distances(self, cluster_centers):

        """Update min distances given cluster centers.

        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        # Update min_distances for all examples given new cluster center.
        x = self.features[cluster_centers]
        dist = pairwise_distances(self.features, x, metric=self.metric)

        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, N):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
          N: batch size

        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        new_batch = []

        for _ in range(N):
            if self.min_distances is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)

            self.update_distances([ind])
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        return new_batch
    
def region_setcover(feat, feat_meta, N):
    res = kCenterGreedy(feat).select_batch_(N)
    res = np.array(feat_meta)[res].tolist()
    return res


def region_clustering(feat, feat_meta, n_clusters):
    print(f"minibatch kmeans, fit")
    res = []
    K_cluster_index = MiniBatchKMeans(n_clusters, reassignment_ratio=0).fit_predict(feat)  # cluster index (n_samples,)

    # select the most uncertain region within each cluster
    for c in range(n_clusters):
        cluster_meta = [m for m in feat_meta if m["id"] in np.where(K_cluster_index==c)[0].tolist()]
        cluster_priorities = [m["uncertainty"] for m in cluster_meta]
        # selected the most uncertain region in the cluster
        if len(cluster_priorities)> 0:
            res.append(cluster_meta[np.argsort(cluster_priorities)[-1]])

    return res
