import torch
import torch.nn as nn

from concept_guided_depth_model import PhaseAwarePooling
from dual_stream_mstcn_detection import FrameEncoder2D, MultiScaleTemporalBlock, TemporalAttentionPooling


class SizeRoutedDepthExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.25, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawSizeRoutedDepthModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 10,
        frame_feature_dim: int = 24,
        temporal_channels: int = 48,
        temporal_blocks: int = 3,
        dropout: float = 0.30,
        num_size_classes: int = 7,
        num_depth_classes: int = 3,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_size_classes = int(num_size_classes)
        self.num_depth_classes = int(num_depth_classes)

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
        self.experts = nn.ModuleList(
            [
                SizeRoutedDepthExpert(
                    in_dim=trunk_dim,
                    hidden_dim=max(48, trunk_dim),
                    dropout=dropout,
                    num_classes=num_depth_classes,
                )
                for _ in range(int(num_size_classes))
            ]
        )
        self.feature_dim = int(trunk_dim)

    @staticmethod
    def compute_delta(x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    def encode(self, raw_window: torch.Tensor, norm_window: torch.Tensor):
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
        return {
            "raw_seq": raw_seq,
            "shape_seq": shape_seq,
            "delta_seq": delta_seq,
            "temporal_seq": temporal_seq,
            "global_feat": global_feat,
            "phase_feat": phase_feat,
            "fused_feat": fused,
            "trunk_feat": trunk_feat,
            "attn_weights": attn_weights,
            "phase_masks": phase_masks,
        }

    def route_logits(self, trunk_feat: torch.Tensor, size_idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            trunk_feat.shape[0],
            self.num_depth_classes,
            dtype=trunk_feat.dtype,
            device=trunk_feat.device,
        )
        for expert_idx, expert in enumerate(self.experts):
            mask = size_idx.long() == int(expert_idx)
            if torch.any(mask):
                logits[mask] = expert(trunk_feat[mask])
        return logits

    def route_logits_soft(self, trunk_feat: torch.Tensor, size_probs: torch.Tensor) -> torch.Tensor:
        expert_logits = []
        for expert in self.experts:
            expert_logits.append(expert(trunk_feat).unsqueeze(1))
        expert_logits = torch.cat(expert_logits, dim=1)  # (B, S, D)
        if size_probs.dim() != 2 or size_probs.shape[1] != self.num_size_classes:
            raise ValueError(
                f"size_probs must have shape (B, {self.num_size_classes}), got {tuple(size_probs.shape)}."
            )
        mixed_logits = torch.sum(expert_logits * size_probs.unsqueeze(-1), dim=1)
        return mixed_logits

    def forward_soft(
        self,
        raw_window: torch.Tensor,
        norm_window: torch.Tensor,
        size_probs: torch.Tensor,
        return_features: bool = False,
    ):
        feats = self.encode(raw_window, norm_window)
        logits = self.route_logits_soft(feats["trunk_feat"], size_probs)
        probs = torch.softmax(logits, dim=1)
        if return_features:
            return logits, probs, feats
        return logits, probs

    def forward(self, raw_window: torch.Tensor, norm_window: torch.Tensor, size_idx: torch.Tensor, return_features: bool = False):
        feats = self.encode(raw_window, norm_window)
        logits = self.route_logits(feats["trunk_feat"], size_idx)
        probs = torch.softmax(logits, dim=1)
        if return_features:
            return logits, probs, feats
        return logits, probs
