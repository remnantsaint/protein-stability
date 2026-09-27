import math
import torch
import torch.nn as nn
import torch.nn.functional as F


AA_FEATURE_DIM = 10


class Predictor(nn.Module):
    def __init__(
        self,
        use_gate: bool = True,
        use_aa_features: bool = True,
        use_dropout: bool = True,
        dropout_rate_input: float = 0.2,
        dropout_rate_conv: float = 0.2,
        dropout_rate_pool: float = 0.2,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.use_aa_features = use_aa_features

        self.conv1d = nn.Conv1d(
            in_channels=1280,
            out_channels=128,
            kernel_size=15,
            padding=7,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.conv_ln = nn.LayerNorm(128)

        self.input_proj = nn.Linear(1280, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            activation="relu",
            batch_first=False,
            norm_first=False,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
        )
        self.trans_ln = nn.LayerNorm(128)

        self.gate_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )

        self.aa_feature_proj = nn.Sequential(
            nn.Linear(AA_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        output_input_dim = 544 if self.use_aa_features else 512
        self.output_layer = nn.Sequential(
            nn.Linear(output_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, wt_embedding, vt_embedding, lengths=None, aa_features=None):
        D = wt_embedding - vt_embedding

        pos_mask = None
        if lengths is not None:
            pos_mask = torch.arange(D.shape[1], device=D.device).unsqueeze(0) < lengths.unsqueeze(1)
            pos_mask_f = pos_mask.unsqueeze(-1).float()
        else:
            pos_mask_f = None

        C_conv = self.conv1d(D.transpose(1, 2))
        C_conv = self.relu(C_conv)
        C_conv_T = self.conv_ln(C_conv.transpose(1, 2))

        T = self.input_proj(D)
        T_transposed = T.transpose(0, 1)
        src_key_padding_mask = None
        if pos_mask is not None:
            src_key_padding_mask = ~pos_mask

        F_transposed = self.transformer_encoder(
            T_transposed,
            src_key_padding_mask=src_key_padding_mask,
        )
        F = self.trans_ln(F_transposed.transpose(0, 1))

        if pos_mask_f is not None:
            F_masked = F * pos_mask_f
            trans_avg = F_masked.sum(dim=1) / (lengths.unsqueeze(1).float() + 1e-8)
            F_masked_max = F.clone()
            F_masked_max = F_masked_max.masked_fill(~pos_mask.unsqueeze(-1), float("-inf"))
            trans_max = F_masked_max.max(dim=1)[0]
        else:
            trans_avg = F.mean(dim=1)
            trans_max = F.max(dim=1)[0]

        P_trans = torch.cat([trans_avg, trans_max], dim=1)

        if pos_mask_f is not None:
            C_masked = C_conv_T * pos_mask_f
            conv_avg = C_masked.sum(dim=1) / (lengths.unsqueeze(1).float() + 1e-8)
            C_masked_max = C_conv_T.clone()
            C_masked_max = C_masked_max.masked_fill(~pos_mask.unsqueeze(-1), float("-inf"))
            conv_max = C_masked_max.max(dim=1)[0]
        else:
            conv_avg = C_conv_T.mean(dim=1)
            conv_max = C_conv_T.max(dim=1)[0]

        P_conv = torch.cat([conv_avg, conv_max], dim=1)

        P_concat = torch.cat([P_conv, P_trans], dim=1)
        if self.use_gate:
            gate = self.gate_fc(P_concat)
            P_fused = torch.cat([gate * P_conv, (1 - gate) * P_trans], dim=1)
        else:
            P_fused = P_concat

        if self.use_aa_features:
            if aa_features is None:
                aa_features = torch.zeros(
                    (P_fused.shape[0], AA_FEATURE_DIM),
                    device=P_fused.device,
                    dtype=P_fused.dtype,
                )
            else:
                aa_features = aa_features.to(device=P_fused.device, dtype=P_fused.dtype)
            aa_emb = self.aa_feature_proj(aa_features)
            final_features = torch.cat([P_fused, aa_emb], dim=1)
        else:
            final_features = P_fused

        return self.output_layer(final_features)
