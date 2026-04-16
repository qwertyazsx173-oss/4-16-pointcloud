import torch
import torch.nn as nn


class CorrespondenceReliability(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=1):
        super(CorrespondenceReliability, self).__init__()

        in_dim = 8  # [mean_feat_dist, mean_xyz_dist, overlap_ratio, ref_valid_ratio, src_valid_ratio, score, logP, sqrtP]

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        ref_node_corr_knn_points,
        src_node_corr_knn_points,
        ref_node_corr_knn_feats,
        src_node_corr_knn_feats,
        ref_node_corr_knn_masks,
        src_node_corr_knn_masks,
        node_corr_scores,
    ):
        # ref/src_node_corr_knn_points: (P, K, 3)
        # ref/src_node_corr_knn_feats : (P, K, C)
        # ref/src_node_corr_knn_masks : (P, K)
        # node_corr_scores            : (P,)

        eps = 1e-8

        ref_masks = ref_node_corr_knn_masks.float()
        src_masks = src_node_corr_knn_masks.float()

        ref_valid_ratio = ref_masks.mean(dim=1, keepdim=True)  # (P, 1)
        src_valid_ratio = src_masks.mean(dim=1, keepdim=True)  # (P, 1)

        pair_mask = ref_masks.unsqueeze(2) * src_masks.unsqueeze(1)  # (P, K, K)
        valid_pair_count = pair_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)  # (P, 1, 1)

        # feature distance
        feat_diff = ref_node_corr_knn_feats.unsqueeze(2) - src_node_corr_knn_feats.unsqueeze(1)  # (P, K, K, C)
        feat_dist = torch.norm(feat_diff, dim=-1)  # (P, K, K)
        mean_feat_dist = (feat_dist * pair_mask).sum(dim=(1, 2), keepdim=True) / valid_pair_count  # (P, 1, 1)

        # geometry distance
        xyz_diff = ref_node_corr_knn_points.unsqueeze(2) - src_node_corr_knn_points.unsqueeze(1)  # (P, K, K, 3)
        xyz_dist = torch.norm(xyz_diff, dim=-1)  # (P, K, K)
        mean_xyz_dist = (xyz_dist * pair_mask).sum(dim=(1, 2), keepdim=True) / valid_pair_count  # (P, 1, 1)

        # rough overlap proxy
        overlap_map = ((xyz_dist < 0.10).float() * pair_mask)  # 0.10 is just a soft heuristic for v1
        overlap_ratio = overlap_map.sum(dim=(1, 2), keepdim=True) / valid_pair_count  # (P, 1, 1)

        score = node_corr_scores.unsqueeze(1)  # (P, 1)
        log_score = torch.log(score.clamp(min=eps))
        sqrt_score = torch.sqrt(score.clamp(min=0.0))

        x = torch.cat(
            [
                mean_feat_dist.squeeze(-1),   # (P, 1)
                mean_xyz_dist.squeeze(-1),    # (P, 1)
                overlap_ratio.squeeze(-1),    # (P, 1)
                ref_valid_ratio,              # (P, 1)
                src_valid_ratio,              # (P, 1)
                score,                        # (P, 1)
                log_score,                    # (P, 1)
                sqrt_score,                   # (P, 1)
            ],
            dim=1,
        )  # (P, 8)

        logits = self.mlp(x).squeeze(1)  # (P,)
        reliability = torch.sigmoid(logits)

        return reliability
