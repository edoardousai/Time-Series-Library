from itertools import combinations
from typing import List

from sparselinear.sparselinear import SparseLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True,
                                          unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class HNN(nn.Module):
    """
    Homological Neural Network architecture for both spatial and temporal
    attention.
    This model is constructed with predefined layers and connectivity
    calculated from `GraphGenerator, which is passed in as init parameters.
    Attributes:
        sl1 (SparseLinear): The first sparse linear layer.
        fc1 (nn.Linear): The first fully connected layer for residual connection.
        sl2 (SparseLinear): The second sparse linear layer.
        fc2 (nn.Linear): The second fully connected for residual connection.
        sl3 (SparseLinear): The third sparse linear layer.
        fc3 (nn.Linear): The third fully connected for residual connection.
        fc4 (nn.Linear): The fourth fully connected layer to read out layer.
        read_out (nn.Linear): The linear readout layer.
        has_l4 (bool): Flag indicating whether the model has a fourth layer, as
            the network might not contain any 4-clique.
    """

    def __init__(
            self,
            network: dict,
            dropout: float,
            out_features: int,
    ):
        """
        Initialize the HNN model.
        Args:
            l1 (int): The number of features in the first layer.
            l2 (int): The number of features in the second layer.
            l3 (int): The number of features in the third layer.
            l4 (int): The number of features in the fourth layer.
            c1 (List[List[int]]): The connectivity matrix for the first layer.
             The shape of the matrix is `[2, N]`.
            c2 (List[List[int]]): The connectivity matrix for the second layer.
             The shape of the matrix is `[2, N]`.
            c3 (List[List[int]]): The connectivity matrix for the third layer.
             The shape of the matrix is `[2, N]`.
            dropout (float): The dropout rate for the model.
        """
        super().__init__()
        l1, l2, l3, l4 = len(network[1]), len(network[2]), len(network[3]), len(
            network[4])
        self.dropout = dropout
        for k, v in network.items():
            network[k] = [tuple(clique) for clique in v]
        self.sl1 = self._get_sparse_layer(l1, l2,
                                          self._get_connection(network[1],
                                                               network[2]))
        self.sl2 = self._get_sparse_layer(l2, l3,
                                          self._get_connection(network[2],
                                                               network[3]))
        self.sl3 = self._get_sparse_layer(l3, l4,
                                          self._get_connection(network[3],
                                                               network[4]))

        self.readout1 = self._get_sparse_layer(l2, l1,
                                               self._get_connection(network[2],
                                                                    network[1]))
        self.readout2 = self._get_sparse_layer(l3, l1,
                                               self._get_connection(network[3],
                                                                    network[1]))
        self.readout3 = self._get_sparse_layer(l4, l1,
                                               self._get_connection(network[4],
                                                                    network[1]))

    def _get_connection(self, prev_cliques, next_cliques):
        connections = [[], []]
        if not prev_cliques or not next_cliques:
            return connections
        if len(prev_cliques[0]) > len(next_cliques[0]):
            return self._get_connection(next_cliques, prev_cliques)[(1, 0), :]
        prev_cliques_index_map = {clique: i for i, clique in
                                  enumerate(prev_cliques)}
        prev_clique_size = len(prev_cliques[0])
        for j, next_clique in enumerate(next_cliques):
            for prev_clique in combinations(next_clique, prev_clique_size):
                i = prev_cliques_index_map.get(prev_clique)
                connections[1].append(i)
                connections[0].append(j)
        return torch.tensor(connections, dtype=torch.int64)

    def _get_sparse_layer(self, l1, l2, connectivity):
        return nn.Sequential(
            SparseLinear(l1, l2, connectivity=connectivity),
            nn.LayerNorm(l2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.
        Args: x (torch.Tensor): The input tensor, typically of shape `[B, P, Q]`.
        The functions are applied on the Q dimension.
        """
        x_s1 = self.sl1(x)
        x_s2 = self.sl2(x_s1)
        x_s3 = self.sl3(x_s2)

        return self.readout1(x_s1) + self.readout2(x_s2) + self.readout3(x_s3)


class Model(nn.Module):
    def __init__(self, configs: argparse.Namespace, network: dict):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(configs.c_out, 96, 1, batch_first=True)
        self.feature_fc = nn.Linear(96, configs.c_out)
        self.time_fc = nn.Linear(96, configs.pred_len)
        # self.hnn = HNN(network, dropout=configs.dropout, out_features=configs.c_out)
        self.rev_in = RevIN(configs.c_out, affine=True)
        self.num_features = configs.c_out
        self.pred_len = configs.pred_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> torch.Tensor:
        x = x_enc
        B, L, N = x.shape
        x = self.rev_in(x, "norm")
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [B, L, 96], apply fc to get feature from hidden states
        x = self.feature_fc(lstm_out)
        # x = self.hnn(x)

        # apply fc to map to pred_len
        x = self.time_fc(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev_in(x, "denorm")
        return x
