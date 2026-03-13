import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder2D(nn.Module):
    def __init__(self, in_channels=1, base_channels=24, out_dim=32, dropout=0.2):
        super().__init__()
        mid_channels = max(base_channels, out_dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_dim)

    def forward(self, x):
        # x: (B,T,1,H,W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.net(x).reshape(b * t, self.out_dim)
        x = self.dropout(x)
        return x.reshape(b, t, self.out_dim)


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 2, 4), dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2) * int(d),
                        dilation=int(d),
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )
                for d in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        branch_outputs = [branch(x) for branch in self.branches]
        fused = self.fuse(torch.cat(branch_outputs, dim=1))
        fused = self.dropout(fused)
        return F.relu(fused + residual, inplace=True)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.attn = nn.Linear(channels, 1)
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = int(channels)

    def forward(self, x):
        # x: (B,T,C)
        weights = torch.softmax(self.attn(x), dim=1)
        attn_feat = torch.sum(weights * x, dim=1)
        mean_feat = torch.mean(x, dim=1)
        max_feat = torch.max(x, dim=1).values
        pooled = torch.cat([attn_feat, mean_feat, max_feat], dim=1)
        return self.proj(pooled), weights


class DualStreamMSTCNDetector(nn.Module):
    def __init__(
        self,
        seq_len=10,
        frame_feature_dim=32,
        temporal_channels=64,
        temporal_blocks=3,
        dropout=0.35,
        use_delta_branch=True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.use_delta_branch = bool(use_delta_branch)

        self.raw_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.25, dropout * 0.5),
        )
        self.delta_encoder = (
            FrameEncoder2D(
                in_channels=1,
                base_channels=max(16, frame_feature_dim),
                out_dim=frame_feature_dim,
                dropout=min(0.25, dropout * 0.5),
            )
            if self.use_delta_branch
            else None
        )

        temporal_input_dim = frame_feature_dim * (2 if self.use_delta_branch else 1)
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
                    dropout=min(0.35, dropout),
                )
                for _ in range(int(temporal_blocks))
            ]
        )
        self.pooling = TemporalAttentionPooling(temporal_channels, dropout=min(0.35, dropout))
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling.out_dim, max(32, temporal_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(32, temporal_channels // 2), 1),
        )
        self.feature_dim = int(self.pooling.out_dim)

    @staticmethod
    def compute_delta(x):
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    def encode_sequence(self, x):
        raw_seq = self.raw_encoder(x)
        streams = [raw_seq]
        delta_seq = None
        if self.use_delta_branch:
            delta_x = self.compute_delta(x)
            delta_seq = self.delta_encoder(delta_x)
            streams.append(delta_seq)

        seq = torch.cat(streams, dim=-1)  # (B,T,C)
        seq = seq.transpose(1, 2)  # (B,C,T)
        seq = self.temporal_input(seq)
        for block in self.temporal_blocks:
            seq = block(seq)
        temporal_seq = seq.transpose(1, 2)  # (B,T,C)
        features, attn_weights = self.pooling(temporal_seq)
        return {
            "raw_seq": raw_seq,
            "delta_seq": delta_seq,
            "temporal_seq": temporal_seq,
            "pooled_features": features,
            "attn_weights": attn_weights,
        }

    def forward(self, x, return_features=False):
        feats = self.encode_sequence(x)
        logit = self.classifier(feats["pooled_features"])
        if return_features:
            return logit, feats
        return logit
