import torch
import torch.nn as nn

from concept_guided_depth_model import PhaseAwarePooling
from dual_stream_mstcn_detection import FrameEncoder2D, MultiScaleTemporalBlock, TemporalAttentionPooling


class RawHybridPositiveSizeModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 10,
        frame_feature_dim: int = 24,
        temporal_channels: int = 48,
        temporal_blocks: int = 3,
        dropout: float = 0.25,
        num_size_classes: int = 7,
        num_tabular_features: int = 19,
        tabular_hidden_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_size_classes = int(num_size_classes)
        self.num_tabular_features = int(num_tabular_features)

        self.amplitude_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.20, dropout * 0.5),
        )
        self.shape_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.20, dropout * 0.5),
        )
        self.delta_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.20, dropout * 0.5),
        )

        temporal_input_dim = frame_feature_dim * 3
        self.temporal_input = nn.Sequential(
            nn.Conv1d(temporal_input_dim, temporal_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                MultiScaleTemporalBlock(
                    channels=temporal_channels,
                    kernel_size=3,
                    dilations=(1, 2, 4),
                    dropout=min(0.30, dropout),
                )
                for _ in range(int(temporal_blocks))
            ]
        )
        self.global_pooling = TemporalAttentionPooling(temporal_channels, dropout=min(0.30, dropout))
        self.phase_pooling = PhaseAwarePooling(temporal_channels, dropout=min(0.30, dropout))

        fused_dim = int(self.global_pooling.out_dim + self.phase_pooling.out_dim)
        trunk_dim = max(64, temporal_channels)
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.tabular_branch = nn.Sequential(
            nn.LayerNorm(int(num_tabular_features)),
            nn.Linear(int(num_tabular_features), int(tabular_hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(tabular_hidden_dim), int(tabular_hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tabular_to_trunk = nn.Linear(int(tabular_hidden_dim), trunk_dim)
        fusion_hidden = max(96, trunk_dim)
        self.fusion = nn.Sequential(
            nn.Linear(trunk_dim * 3, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.size_cls_head = nn.Sequential(
            nn.Linear(fusion_hidden, max(64, fusion_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(64, fusion_hidden), int(num_size_classes)),
        )
        self.size_ord_head = nn.Sequential(
            nn.Linear(fusion_hidden, max(48, fusion_hidden // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(48, fusion_hidden // 2), int(num_size_classes - 1)),
        )
        self.size_residual_head = nn.Sequential(
            nn.Linear(fusion_hidden, max(64, fusion_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(64, fusion_hidden), 1),
        )
        self.feature_dim = int(fusion_hidden)

    @staticmethod
    def compute_delta(x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    def encode(self, raw_window: torch.Tensor, norm_window: torch.Tensor, tabular_x: torch.Tensor):
        raw_seq = self.amplitude_encoder(raw_window)
        shape_seq = self.shape_encoder(norm_window)
        delta_seq = self.delta_encoder(self.compute_delta(norm_window))
        seq = torch.cat([raw_seq, shape_seq, delta_seq], dim=-1).transpose(1, 2)
        seq = self.temporal_input(seq)
        for block in self.temporal_blocks:
            seq = block(seq)
        temporal_seq = seq.transpose(1, 2)
        global_feat, attn_weights = self.global_pooling(temporal_seq)
        phase_feat, phase_masks = self.phase_pooling(temporal_seq, raw_window)
        fused = torch.cat([global_feat, phase_feat], dim=1)
        trunk_feat = self.trunk(fused)

        tabular_feat = self.tabular_branch(tabular_x)
        tabular_proj = self.tabular_to_trunk(tabular_feat)
        hybrid_feat = self.fusion(torch.cat([trunk_feat, tabular_proj, trunk_feat * tabular_proj], dim=1))
        return {
            "temporal_seq": temporal_seq,
            "global_feat": global_feat,
            "phase_feat": phase_feat,
            "trunk_feat": trunk_feat,
            "tabular_feat": tabular_feat,
            "tabular_proj": tabular_proj,
            "hybrid_feat": hybrid_feat,
            "attn_weights": attn_weights,
            "phase_masks": phase_masks,
        }

    def forward(self, raw_window: torch.Tensor, norm_window: torch.Tensor, tabular_x: torch.Tensor, return_features: bool = False):
        feats = self.encode(raw_window, norm_window, tabular_x)
        size_logits = self.size_cls_head(feats["hybrid_feat"])
        size_ord_logits = self.size_ord_head(feats["hybrid_feat"])
        size_probs = torch.softmax(size_logits, dim=1)
        if size_probs.shape[1] != self.num_size_classes:
            raise ValueError("Unexpected number of size classes in logits.")
        size_values = torch.linspace(0.0, 1.0, steps=self.num_size_classes, device=size_probs.device, dtype=size_probs.dtype)
        expected_norm = torch.sum(size_probs * size_values.view(1, -1), dim=1, keepdim=True)
        residual = 0.35 * torch.tanh(self.size_residual_head(feats["hybrid_feat"]))
        size_reg_norm = torch.clamp(expected_norm + residual, 0.0, 1.0)
        if return_features:
            return size_logits, size_ord_logits, size_reg_norm, size_probs, feats
        return size_logits, size_ord_logits, size_reg_norm, size_probs
