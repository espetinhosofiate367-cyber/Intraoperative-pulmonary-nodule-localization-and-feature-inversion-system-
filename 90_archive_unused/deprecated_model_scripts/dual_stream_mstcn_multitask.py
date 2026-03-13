import torch.nn as nn

from dual_stream_mstcn_detection import DualStreamMSTCNDetector


class DualStreamMSTCNMultiTask(DualStreamMSTCNDetector):
    def __init__(
        self,
        seq_len=10,
        frame_feature_dim=32,
        temporal_channels=64,
        temporal_blocks=3,
        dropout=0.35,
        use_delta_branch=True,
        num_size_classes=7,
        num_depth_classes=3,
    ):
        super().__init__(
            seq_len=seq_len,
            frame_feature_dim=frame_feature_dim,
            temporal_channels=temporal_channels,
            temporal_blocks=temporal_blocks,
            dropout=dropout,
            use_delta_branch=use_delta_branch,
        )
        hidden_dim = max(32, self.feature_dim // 2)
        self.size_cls_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(num_size_classes)),
        )
        self.size_reg_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.depth_coarse_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(num_depth_classes)),
        )
        self.num_size_classes = int(num_size_classes)
        self.num_depth_classes = int(num_depth_classes)

    def forward(self, x, return_features=False):
        feats = self.encode_sequence(x)
        pooled = feats["pooled_features"]
        det_logit = self.classifier(pooled)
        size_logits = self.size_cls_head(pooled)
        size_reg = self.size_reg_head(pooled)
        depth_logits = self.depth_coarse_head(pooled)
        if return_features:
            return det_logit, size_logits, size_reg, depth_logits, feats
        return det_logit, size_logits, size_reg, depth_logits
