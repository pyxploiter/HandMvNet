import math
import torch
from torch import nn
from einops import rearrange


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, kernel_size=1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=4):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
    
    def forward(self, x):
        # Create frequency components for sinusoids
        frequencies = torch.exp(torch.arange(0, self.num_frequencies, 2) * (-torch.log(torch.tensor(10000.0)) / self.num_frequencies))
        frequencies = frequencies.unsqueeze(0).unsqueeze(0).to(x.device)
        
        # Calculate sine and cosine components for batch of x
        encoding = torch.cat([torch.sin(x.unsqueeze(-1) * frequencies),
                              torch.cos(x.unsqueeze(-1) * frequencies)], dim=-1)

        return encoding.flatten(start_dim=-2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1, max_len, d_model)

        # Check for even d_model
        if d_model % 2 == 0:
            self.pe[0, :, 0::2] = torch.sin(position * div_term)
            self.pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            # Handle odd d_model
            self.pe[0, :, 0::2] = torch.sin(position * div_term)
            self.pe[0, :, 1::2] = torch.cos(position * div_term[:-1])

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dim_head=128, dropout=0., custom_query_length=0, use_query_tokens_from_start=True):
        super().__init__()

        self.custom_query_length = custom_query_length
        self.use_query_tokens_from_start = use_query_tokens_from_start
        inner_dim = dim_head * n_heads

        self.heads = n_heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_k = nn.Linear(d_model, inner_dim, bias=False)
        self.to_v = nn.Linear(d_model, inner_dim, bias=False)
        # make output dim same as d_model
        self.to_out = nn.Linear(inner_dim, d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        # self.dropout2 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(dim=d_model, hidden_dim=dim_head, dropout=dropout)

    def forward(self, x, return_attention=False):
        # x: [batch_size, seq_len, d_model]
        if self.custom_query_length > 0:
            if self.use_query_tokens_from_start:
                # use first 21 tokens as query, rest as key and value
                _q, _k, _v = x[:, :self.custom_query_length, :], x[:, self.custom_query_length:, :], x[:, self.custom_query_length:, :]
            else:
                # use last 21 tokens as query, rest as key and value
                _q, _k, _v = x[:, -self.custom_query_length:, :], x[:, :-self.custom_query_length, :], x[:, :-self.custom_query_length, :]
        else:
            _q, _k, _v = x, x, x
        q = rearrange(self.to_q(_q), "b i (h d) -> b h i d", h=self.heads)
        k = rearrange(self.to_k(_k), "b i (h d) -> b h i d", h=self.heads)
        v = rearrange(self.to_v(_v), "b i (h d) -> b h i d", h=self.heads)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # Apply softmax and obtain weighted sum
        attn = self.attend(dots)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout1(out)

        # add & norm
        out = self.norm1(out + _q)
        _out = out
        # feed forward
        out = self.ff(out)
        # add & norm
        out = self.norm2(out + _out)

        if return_attention:
            return out, attn
        return out


class MultiHeadAttentionLearnableQuery(nn.Module):
    def __init__(self, inp, oup, max_tokens, heads=8, dim_head=256, dropout=0., cross_attn=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.cross_attn = cross_attn

        self.heads = heads
        self.scale = dim_head ** -0.5

        if self.cross_attn:
            self.probe = nn.Parameter(torch.randn(1, 21, inp))

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.ff = FeedForward(dim=oup, hidden_dim=dim_head, dropout=dropout)
        self.pos_embed = PositionalEncoding(d_model=inp, max_len=max_tokens)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, feat_dim = x.size()
        # Add positional embedding
        x = self.pos_embed(x)

        if self.cross_attn:
            probe = self.probe.repeat((batch_size, 1, 1))
            probe = self.pos_embed(probe)

        if self.cross_attn:
            q = rearrange(self.to_q(probe), "b i (h d) -> b h i d", h=self.heads)
        else:
            q = rearrange(self.to_q(x), "b i (h d) -> b h i d", h=self.heads)
        k = rearrange(self.to_k(x), "b i (h d) -> b h i d", h=self.heads)
        v = rearrange(self.to_v(x), "b i (h d) -> b h i d", h=self.heads)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # Apply softmax and obtain weighted sum
        attn = self.attend(dots)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # Reshape and apply output layer
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if self.cross_attn:
            out = self.ff(out) + out
        else:
            out = out + x
            out = self.ff(out) + out

        if return_attention:
            return out, attn
        return out


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class GraphPool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphPool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class GraphUnpool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphUnpool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        # print(in_c, out_c)
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph.to(inputs.device), self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]

        # print(mul_L.shape, inputs.shape)
        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]

        # print(result.shape, self.weight.shape)
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)

        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X
