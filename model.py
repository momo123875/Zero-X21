import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import encoder
import decoder
import utils
from einops import rearrange


class TransformerPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        K = x.size(1)
        position = torch.arange(0, K, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float, device=x.device)
            * (-torch.log(torch.tensor(10000.0)) / self.dim)
        )
        pe = torch.zeros(K, self.dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).expand(x.size(0), -1, -1)


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        attn_output, _ = self.attention(q, k, v)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x


class ImprovedMLPDynamicWeightFusion(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super(ImprovedMLPDynamicWeightFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, feature_vector, attended_feature):
        combined_features = torch.cat([feature_vector, attended_feature], dim=-1)
        alpha = self.mlp(combined_features)
        fused_feature = alpha * feature_vector + (1 - alpha) * attended_feature
        return fused_feature


class SimFeatureSearch(torch.nn.Module):
    def __init__(self, feature_dim, top_k=5):
        super(SimFeatureSearch, self).__init__()
        self.top_k = top_k
        self.feature_dim = feature_dim

    def forward(self, feature_map):
        N, C, D, H, W = feature_map.shape
        if D <= H and D <= W:
            slices = feature_map.permute(2, 0, 1, 3, 4).reshape(D, N, C, H * W)
            slice_dim = 0
            output_shape = (N, C, D, H, W)
        elif H <= D and H <= W:
            slices = feature_map.permute(3, 0, 1, 2, 4).reshape(H, N, C, D * W)
            slice_dim = 1
            output_shape = (N, C, H, D, W)
        else:
            slices = feature_map.permute(4, 0, 1, 2, 3).reshape(W, N, C, D * H)
            slice_dim = 2
            output_shape = (N, C, W, D, H)
        weighted_features = []
        for slice_features in slices:
            similarity_matrix = torch.matmul(
                slice_features.transpose(1, 2), slice_features
            )
            _, topk_indices = torch.topk(similarity_matrix, self.top_k, dim=-1)
            topk_features = torch.gather(
                slice_features.unsqueeze(-1).expand(-1, -1, -1, self.top_k),
                2,
                topk_indices.unsqueeze(1).expand(-1, C, -1, -1),
            )
            distances = torch.abs(
                topk_indices
                - torch.arange(topk_indices.size(1), device=topk_indices.device).view(
                    1, -1, 1
                )
            )
            distances = distances + 1e-5
            weights = 1.0 / distances
            weights = weights / weights.sum(dim=-1, keepdim=True)
            weighted_avg_features = (topk_features * weights.unsqueeze(1)).sum(dim=-1)
            weighted_features.append(weighted_avg_features)
        weighted_features = torch.stack(weighted_features, dim=0).view(output_shape)
        if slice_dim == 1:
            weighted_features = weighted_features.permute(0, 1, 3, 2, 4)
        elif slice_dim == 2:
            weighted_features = weighted_features.permute(0, 1, 3, 4, 2)
        return weighted_features


class AttentionGuidedInterpolation(nn.Module):
    def __init__(self, feature_dim, num_heads, D_hr=80, H_hr=80, W_hr=80, r=1):
        super(AttentionGuidedInterpolation, self).__init__()
        self.simFeatureS = SimFeatureSearch(feature_dim)
        self.D_hr = D_hr
        self.H_hr = H_hr
        self.W_hr = W_hr
        self.r = r
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def compute_relative_distances(self, xyz_hr, neighbor_coords):
        xyz_hr_expanded = xyz_hr.unsqueeze(2).unsqueeze(2)
        relative_distances = torch.norm(xyz_hr_expanded - neighbor_coords, dim=-1)
        return relative_distances

    def find_neighbor_coords(self, xyz_hr, feature_map_shape):
        N, K, _ = xyz_hr.shape
        D, H, W = feature_map_shape[-3:]
        xyz = (xyz_hr + 1) / 2
        xyz[..., 0] *= D - 1
        xyz[..., 1] *= H - 1
        xyz[..., 2] *= W - 1
        grid_coord_k = xyz.floor().long()
        min_dim_index = torch.argmin(torch.tensor([D, H, W])).item()
        rd, rh, rw = 2 / D, 2 / H, 2 / W
        dd = torch.linspace(-self.r, self.r, 2 * self.r + 1).to(xyz.device) * rd
        dh = torch.linspace(-self.r, self.r, 2 * self.r + 1).to(xyz.device) * rh
        dv = torch.linspace(-self.r, self.r, 2 * self.r + 1).to(xyz.device) * rw
        if min_dim_index == 0:
            delta_2d = torch.stack(torch.meshgrid(dh, dv, indexing="ij"), dim=-1).view(
                1, 1, -1, 2
            )
            neighbor_coords_2d = grid_coord_k[..., 1:].unsqueeze(2) + delta_2d
            d_coord = (
                grid_coord_k[..., 0:1]
                .unsqueeze(2)
                .expand(-1, -1, neighbor_coords_2d.shape[2], -1)
            )
            neighbor_coords = torch.cat([d_coord, neighbor_coords_2d], dim=-1)
        elif min_dim_index == 1:
            delta_2d = torch.stack(torch.meshgrid(dd, dv, indexing="ij"), dim=-1).view(
                1, 1, -1, 2
            )
            neighbor_coords_2d = grid_coord_k[..., ::2].unsqueeze(2) + delta_2d
            h_coord = (
                grid_coord_k[..., 1:2]
                .unsqueeze(2)
                .expand(-1, -1, neighbor_coords_2d.shape[2], -1)
            )
            neighbor_coords = torch.cat(
                [neighbor_coords_2d[..., :1], h_coord, neighbor_coords_2d[..., 1:]],
                dim=-1,
            )
        else:
            delta_2d = torch.stack(torch.meshgrid(dd, dh, indexing="ij"), dim=-1).view(
                1, 1, -1, 2
            )
            neighbor_coords_2d = grid_coord_k[..., :2].unsqueeze(2) + delta_2d
            w_coord = (
                grid_coord_k[..., 2:3]
                .unsqueeze(2)
                .expand(-1, -1, neighbor_coords_2d.shape[2], -1)
            )
            neighbor_coords = torch.cat([neighbor_coords_2d, w_coord], dim=-1)
        neighbor_coords[..., 0] = (neighbor_coords[..., 0] / (D - 1)) * 2 - 1
        neighbor_coords[..., 1] = (neighbor_coords[..., 1] / (H - 1)) * 2 - 1
        neighbor_coords[..., 2] = (neighbor_coords[..., 2] / (W - 1)) * 2 - 1
        return neighbor_coords

    def forward(self, feature_map, xyz_hr):
        similar_features = self.simFeatureS(feature_map)
        initial_feature_vector = F.grid_sample(
            feature_map,
            xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
        )[:, :, 0, 0, :].permute(0, 2, 1)
        neighbor_coords = self.find_neighbor_coords(
            xyz_hr, feature_map.shape
        ).unsqueeze(2)
        N, K, _, r_area, _ = neighbor_coords.shape
        neighbor_coords_reshaped = neighbor_coords.view(N, K * r_area, 1, 1, 3).flip(-1)
        neighbor_feature_vectors = F.grid_sample(
            feature_map,
            neighbor_coords_reshaped,
            mode="nearest",
            align_corners=False,
        ).view(N, K, r_area, feature_map.size(1))
        similar_feature_vectors = F.grid_sample(
            similar_features,
            neighbor_coords_reshaped,
            mode="nearest",
            align_corners=False,
        ).view(N, K, r_area, feature_map.size(1))
        relative_distances = self.compute_relative_distances(xyz_hr, neighbor_coords)
        relative_weights = 1 / (relative_distances + 1e-6)
        relative_weights /= relative_weights.sum(dim=-1, keepdim=True)
        relative_weights = relative_weights.permute(0, 1, 3, 2)
        weighted_neighbor_features = (neighbor_feature_vectors * relative_weights).sum(
            dim=2
        )
        weighted_similar_features = (similar_feature_vectors * relative_weights).sum(
            dim=2
        )
        combined_features = (weighted_similar_features + weighted_neighbor_features) / 2
        query = self.query(initial_feature_vector)
        key = self.key(combined_features)
        value = self.value(combined_features)
        attn_output, _ = self.attention(query, key, value)
        enhanced_feature_vector = attn_output + initial_feature_vector
        return enhanced_feature_vector


class ArSSR(nn.Module):
    def __init__(
        self,
        encoder_name,
        decoder_name,
        feature_dim,
        decoder_depth,
        decoder_width,
        fourier_dim,
        num_heads,
        fusion_hidden_dim,
    ):
        super(ArSSR, self).__init__()
        if encoder_name == "RDN":
            self.encoder = encoder.RDN(feature_dim=feature_dim)
        elif encoder_name == "SRResnet":
            self.encoder = encoder.SRResnet(feature_dim=feature_dim)
        elif encoder_name == "ResCNN":
            self.encoder = encoder.ResCNN(feature_dim=feature_dim)
        else:
            raise ValueError("Invalid encoder name")
        if decoder_name == "MLP":
            self.decoder = decoder.MLP(
                in_dim=feature_dim + fourier_dim,
                out_dim=1,
                depth=decoder_depth,
                width=decoder_width,
            )
        elif decoder_name == "SIREN":
            self.decoder = decoder.SIREN(
                in_dim=feature_dim + fourier_dim,
                out_dim=1,
                depth=decoder_depth,
                width=decoder_width,
            )
        else:
            raise ValueError("Invalid decoder name")
        self.positional_encoding = TransformerPositionalEncoding(fourier_dim)
        self.cross_attention = CrossAttention(
            d_model=feature_dim + fourier_dim, num_heads=num_heads
        )
        self.fusion = ImprovedMLPDynamicWeightFusion(
            dim=feature_dim + fourier_dim, hidden_dim=fusion_hidden_dim
        )
        self.ati = AttentionGuidedInterpolation(feature_dim, num_heads)

    def forward(self, img_lr_1, xyz_hr_1, img_lr_2, xyz_hr_2):
        feature_map_1 = self.encoder(img_lr_1)
        feature_map_2 = self.encoder(img_lr_2)
        feature_vector_1 = self.ati(feature_map_1, xyz_hr_1)
        feature_vector_2 = self.ati(feature_map_2, xyz_hr_2)
        pos_enc_1 = self.positional_encoding(xyz_hr_1)
        pos_enc_2 = self.positional_encoding(xyz_hr_2)
        feature_vector_1 = torch.cat([feature_vector_1, pos_enc_1], dim=-1)
        feature_vector_2 = torch.cat([feature_vector_2, pos_enc_2], dim=-1)
        feature_vector_1 = self.cross_attention(feature_vector_1, feature_vector_2)
        feature_vector_2 = self.cross_attention(feature_vector_2, feature_vector_1)
        fused_feature_1 = self.fusion(feature_vector_1, feature_vector_2)
        fused_feature_2 = self.fusion(feature_vector_2, feature_vector_1)
        N_1, K_1 = xyz_hr_1.shape[:2]
        N_2, K_2 = xyz_hr_2.shape[:2]
        intensity_pre_1 = self.decoder(fused_feature_1.view(N_1 * K_1, -1)).view(
            N_1, K_1, -1
        )
        intensity_pre_2 = self.decoder(fused_feature_2.view(N_2 * K_2, -1)).view(
            N_2, K_2, -1
        )
        return intensity_pre_1, intensity_pre_2
