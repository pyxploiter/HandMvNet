import torch
import torch.nn as nn

from models.layers import MultiHeadAttention, MultiHeadAttentionLearnableQuery, PositionalEncoding


class CrossAttentionFusion(nn.Module):
    # [b, num_views*21, feat_dim] -> [b, num_views, feat_dim]
    def __init__(self, feat_dim, max_tokens, custom_query_length, num_layers, drop_out=0.):
        super().__init__()
        assert num_layers % 2 == 1, "num_layers must be an odd number"
        num_layers -= 1
        self.feat_dim = feat_dim
        self.pos_encoding = PositionalEncoding(d_model=self.feat_dim, max_len=max_tokens)

        half_layers = num_layers // 2
        layers = [MultiHeadAttention(d_model=self.feat_dim, dropout=drop_out) for _ in range(half_layers)]
        # cross attention using first "custom_query_length" tokens as query and rest as key & value
        layers.append(MultiHeadAttention(d_model=self.feat_dim, dropout=drop_out, custom_query_length=custom_query_length))
        # print("Warning: Replacing cross-attention with self-attention")
        # layers.append(MultiHeadAttention(d_model=self.feat_dim, dropout=drop_out))
        layers.extend([MultiHeadAttention(d_model=self.feat_dim, dropout=drop_out) for _ in range(half_layers)])

        self.attn_fusion = nn.Sequential(*layers)

    def forward(self, x, add_pos=True):
        if add_pos:
            x = self.pos_encoding(x)
        x = self.attn_fusion(x)
        return x


class CrossAttentionFusionLearnableQuery(nn.Module):
    # [b, num_views*21, feat_dim] -> [b, num_views, feat_dim]
    def __init__(self, feat_dim, max_tokens, drop_out=0.):
        super().__init__()
        self.feat_dim = feat_dim

        self.attn_fusion = torch.nn.Sequential(
            MultiHeadAttentionLearnableQuery(inp=self.feat_dim, oup=self.feat_dim, max_tokens=max_tokens, dropout=drop_out, cross_attn=False),
            MultiHeadAttentionLearnableQuery(inp=self.feat_dim, oup=self.feat_dim, max_tokens=max_tokens, dropout=drop_out, cross_attn=False),
            MultiHeadAttentionLearnableQuery(inp=self.feat_dim, oup=self.feat_dim, max_tokens=max_tokens, dropout=drop_out, cross_attn=True),
            MultiHeadAttentionLearnableQuery(inp=self.feat_dim, oup=self.feat_dim, max_tokens=max_tokens, dropout=drop_out, cross_attn=False),
            MultiHeadAttentionLearnableQuery(inp=self.feat_dim, oup=self.feat_dim, max_tokens=max_tokens, dropout=drop_out, cross_attn=False)
        )

    def forward(self, x):
        x = self.attn_fusion(x)
        return x