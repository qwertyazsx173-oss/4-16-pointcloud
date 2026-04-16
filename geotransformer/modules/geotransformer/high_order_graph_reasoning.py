import torch
import torch.nn as nn
import torch.nn.functional as F


class HighOrderGraphReasoning(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=1,
        topk=64,
        graph_k=8,
        compatibility_sigma=0.10,
        min_score=1e-6,
    ):
        super(HighOrderGraphReasoning, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.topk = topk
        self.graph_k = graph_k
        self.compatibility_sigma = compatibility_sigma
        self.min_score = min_score

        # node scalar features:
        # [score, log_score, feat_cos, feat_l2]
        self.node_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # edge input:
        # [h_j, compatibility, residual]
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ref_node_corr_indices,
        src_node_corr_indices,
        node_corr_scores,
        ref_points_c,
        src_points_c,
        ref_feats_c,
        src_feats_c,
    ):
        # inputs:
        # ref_node_corr_indices: (P,)
        # src_node_corr_indices: (P,)
        # node_corr_scores:      (P,)
        # ref_points_c/src_points_c: (N, 3)
        # ref_feats_c/src_feats_c:   (N, C)

        if ref_node_corr_indices.numel() == 0:
            return ref_node_corr_indices, src_node_corr_indices, node_corr_scores

        P = ref_node_corr_indices.shape[0]
        keep = min(self.topk, P)

        # top-M candidate filtering by original coarse score
        top_scores, top_ids = torch.topk(node_corr_scores, k=keep, largest=True, sorted=False)
        ref_idx = ref_node_corr_indices[top_ids]
        src_idx = src_node_corr_indices[top_ids]

        ref_pts = ref_points_c[ref_idx]   # (M, 3)
        src_pts = src_points_c[src_idx]   # (M, 3)
        ref_f = ref_feats_c[ref_idx]      # (M, C)
        src_f = src_feats_c[src_idx]      # (M, C)

        if keep == 1:
            return ref_idx, src_idx, top_scores

        k = min(self.graph_k, keep - 1)
        if k <= 0:
            return ref_idx, src_idx, top_scores

        # node features
        feat_cos = F.cosine_similarity(ref_f, src_f, dim=-1, eps=1e-8).unsqueeze(1)  # (M, 1)
        feat_l2 = torch.norm(ref_f - src_f, dim=-1, keepdim=True)                     # (M, 1)
        score = top_scores.clamp(min=self.min_score).unsqueeze(1)                      # (M, 1)
        log_score = torch.log(score.clamp(min=self.min_score))                         # (M, 1)

        node_x = torch.cat([score, log_score, feat_cos, feat_l2], dim=1)              # (M, 4)
        h = self.node_mlp(node_x)                                                      # (M, H)

        # sparse graph by kNN in reference-side center space
        with torch.no_grad():
            ref_dist_mat = torch.cdist(ref_pts, ref_pts, p=2)                          # (M, M)
            knn_ids = torch.topk(ref_dist_mat, k=k + 1, largest=False).indices[:, 1:] # (M, k)

        for _ in range(self.num_layers):
            ref_nbr_pts = ref_pts[knn_ids]                                             # (M, k, 3)
            src_nbr_pts = src_pts[knn_ids]                                             # (M, k, 3)
            h_nbr = h[knn_ids]                                                         # (M, k, H)

            ref_edge_len = torch.norm(ref_pts.unsqueeze(1) - ref_nbr_pts, dim=-1)     # (M, k)
            src_edge_len = torch.norm(src_pts.unsqueeze(1) - src_nbr_pts, dim=-1)     # (M, k)

            residual = torch.abs(ref_edge_len - src_edge_len)                          # (M, k)
            compatibility = torch.exp(
                - (residual ** 2) / (2.0 * (self.compatibility_sigma ** 2) + 1e-8)
            )                                                                          # (M, k)

            edge_input = torch.cat(
                [h_nbr, compatibility.unsqueeze(-1), residual.unsqueeze(-1)],
                dim=-1,
            )                                                                          # (M, k, H+2)

            msg = self.edge_mlp(edge_input)                                            # (M, k, H)
            msg = msg * compatibility.unsqueeze(-1)                                    # weighted messages
            agg = msg.mean(dim=1)                                                      # (M, H)

            h = h + self.update_mlp(torch.cat([h, agg], dim=1))                        # residual update

        gate = torch.sigmoid(self.out_mlp(h).squeeze(1))                               # (M,)

        with torch.no_grad():
            ref_nbr_pts = ref_pts[knn_ids]
            src_nbr_pts = src_pts[knn_ids]
            ref_edge_len = torch.norm(ref_pts.unsqueeze(1) - ref_nbr_pts, dim=-1)
            src_edge_len = torch.norm(src_pts.unsqueeze(1) - src_nbr_pts, dim=-1)
            residual = torch.abs(ref_edge_len - src_edge_len)
            compatibility = torch.exp(
                - (residual ** 2) / (2.0 * (self.compatibility_sigma ** 2) + 1e-8)
            )
            mean_compat = compatibility.mean(dim=1)                                    # (M,)

        refined_scores = top_scores.clamp(min=self.min_score) * (0.5 * gate + 0.5 * mean_compat)
        refined_scores = refined_scores.clamp(min=self.min_score)

        # sort descending after refinement
        sorted_ids = torch.argsort(refined_scores, descending=True)
        ref_idx = ref_idx[sorted_ids]
        src_idx = src_idx[sorted_ids]
        refined_scores = refined_scores[sorted_ids]

        return ref_idx, src_idx, refined_scores
