import torch.nn as nn


class GeometryGuidedPoseRefinement(nn.Module):
    def __init__(self, num_steps=1):
        super(GeometryGuidedPoseRefinement, self).__init__()
        self.num_steps = num_steps

    def forward(
        self,
        ref_corr_points,
        src_corr_points,
        corr_scores,
        estimated_transform,
    ):
        return estimated_transform
