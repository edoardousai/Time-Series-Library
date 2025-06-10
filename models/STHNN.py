from itertools import combinations

from sparselinear.sparselinear import SparseLinear
import torch
import torch.nn as nn
import argparse

from layers.StandardNorm import Normalize


class HNN(nn.Module):
    """
    Homological Neural Network architecture for both spatial and temporal
    attention.
    This model is constructed with predefined layers and connectivity
    calculated from `GraphGenerator, which is passed in as init parameters.
    Attributes:
        sl1 (SparseLinear): The first sparse linear layer.
        sl2 (SparseLinear): The second sparse linear layer.
        sl3 (SparseLinear): The third sparse linear layer.
        readout1 (SparseLinear): The first readout layer.
        readout2 (SparseLinear): The second readout layer.
        readout3 (SparseLinear): The third readout layer.
    """

    def __init__(self, network: dict, dropout: float):
        """
        Initialize the HNN model.
        Args:
            network (dict): The network structure of TMFG.
            dropout (float): The dropout rate to be applied to the layers.
        """
        super().__init__()
        l1, l2, l3, l4 = (
            len(network[1]),
            len(network[2]),
            len(network[3]),
            len(network[4]),
        )
        self.dropout = dropout
        for k, v in network.items():
            network[k] = [tuple(clique) for clique in v]
        self.sl1 = self._get_sparse_layer(
            l1, l2, self._get_connection(network[1], network[2])
        )
        self.sl2 = self._get_sparse_layer(
            l2, l3, self._get_connection(network[2], network[3])
        )
        self.sl3 = self._get_sparse_layer(
            l3, l4, self._get_connection(network[3], network[4])
        )

        self.readout1 = self._get_sparse_layer(
            l2, l1, self._get_connection(network[2], network[1])
        )
        self.readout2 = self._get_sparse_layer(
            l3, l1, self._get_connection(network[3], network[1])
        )
        self.readout3 = self._get_sparse_layer(
            l4, l1, self._get_connection(network[4], network[1])
        )

    def _get_connection(self, prev_cliques, next_cliques):
        connections = [[], []]
        if not prev_cliques or not next_cliques:
            return connections
        if len(prev_cliques[0]) > len(next_cliques[0]):
            return self._get_connection(next_cliques, prev_cliques)[(1, 0), :]
        prev_cliques_index_map = {clique: i for i, clique in enumerate(prev_cliques)}
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
    """
    This model consists of a single HNN unit, which is constructed by a flatten
    cross correlation matrix. It is designed to take historical multivariate time
    series data as input and produce future predictions.
    Attributes:
        rev_in (RevIN): The reversible input normalization layer.
        hnn (HNN): The HNN layer.
        fc (nn.Sequential): The fully connected layer for residual connection
        of original input to projection layer.
        projection (nn.Linear): The linear projection layer for final prediction.
    """

    def __init__(
        self,
        configs: argparse.Namespace,
        network: dict,
    ):
        """
        Initialize the HNNMixer model.
        Args:
            configs (argparse.Namespace): The configuration object containing
            model parameters.
            network (dict): The network structure of TMFG.
        """
        super().__init__()
        self.prediction_seq_len = configs.pred_len
        self.rev_in = Normalize(configs.c_out, affine=True)
        self.hnn = HNN(network, dropout=configs.dropout)
        self.fc = nn.Sequential(
            nn.Linear(configs.seq_len, configs.seq_len),
            nn.LayerNorm(configs.seq_len),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(
        self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None
    ) -> torch.Tensor:
        """
        Perform a forward pass through the network.
        Args:
            x (torch.Tensor): The input tensor, typically of shape `[B, L, N]`.
            x_mark_enc, x_dec, x_mark_dec (torch.Tensor): Are included for TSLib
            compatibility, but are not used in our model.
        Returns:
            torch.Tensor: The output prediction tensor, typically of shape
            `[B, F, N]`.
        """
        B, L, N = x.shape
        x0 = self.rev_in(x, "norm")

        x = x0.flatten(start_dim=1)  # [B, N, L] -> [B, L * N]

        x = self.hnn(x)

        x = x.view(B, L, N)  # [B, L * N] -> [B, L, N]
        x = x.transpose(1, 2)
        x = self.projection(x + self.fc(x0.transpose(1, 2)))
        x = x.transpose(1, 2)
        x = self.rev_in(x, "denorm")
        return x
