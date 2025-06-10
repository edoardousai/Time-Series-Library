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

class HCNN(nn.Module):
    """
    This model consists of a single HNN unit, which is constructed by a flatten cross correlation matrix.

    It is designed to take historical multivariate time series data as input
    and produce future predictions.

    Attributes:
        rev_in (RevIN): The reversible input normalization layer.
        HNN (HNN): The HNN layer.
        projection (nn.Linear): The linear projection layer.
    """

    def __init__(
        self,
        configs: argparse.Namespace,
        network: dict,
    ):
        """
        Initialize the HNNMixer model.

        Args:
            history_seq_len (int): The length of the input history sequence.
            prediction_seq_len (int): The length of the output prediction sequence.
            spatial (dict): The configuration dictionary for the spatial HNN.
            temporal (dict): The configuration dictionary for the temporal HNN.
            n (int): The number of nodes in the graph.
            dropout (float): The dropout rate for the model.
        """
        super().__init__()
        self.prediction_seq_len = configs.pred_len
        self.tetrahedra = network[4]
        self.triangle = network[3]
        self.edge = network[2]

        self.flatten_tetrahedra = torch.tensor([idx for clique in self.tetrahedra for idx in clique])
        self.flatten_triangle = torch.tensor([idx for clique in self.triangle for idx in clique])
        self.flatten_edge = torch.tensor([idx for clique in self.edge for idx in clique])

        self.rev_in = RevIN(configs.c_out, affine=True)

        filter1_size = configs.filter1_size
        dropout = configs.dropout
        self.conv_tetrahedra = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=filter1_size, kernel_size=4, stride=4
            ),
            nn.InstanceNorm1d(filter1_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=0.01),

            # Flatten the 3D (N, C, L) tensor to 2D (N, C*L)
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(filter1_size * (len(self.tetrahedra)), configs.c_out),
        )

        self.conv_triangle = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=filter1_size, kernel_size=3, stride=3
            ),
            nn.InstanceNorm1d(filter1_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=0.01),
            # Flatten the 3D (N, C, L) tensor to 2D (N, C*L)
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(filter1_size * (len(self.triangle)), configs.c_out),
        )

        self.conv_edge = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=filter1_size, kernel_size=2, stride=2
            ),
            nn.InstanceNorm1d(filter1_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=0.01),
            # Flatten the 3D (N, C, L) tensor to 2D (N, C*L)
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(filter1_size * (len(self.edge)), configs.c_out),
        )
        self.readout = nn.Linear(configs.c_out, configs.c_out)

    def forward(self, x) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            history_data (torch.Tensor): A tensor containing historical data,
             typically of shape `[B, L, N, C]`.
            future_data (torch.Tensor): A tensor containing future data,
             typically of shape `[B, L, N, C]`.
            batch_seen (int): The number of batches seen so far during training.
            epoch (int): The current epoch number.
            train (bool): Flag indicating whether the model is in training mode.

        Returns:
            torch.Tensor: The output prediction tensor, typically of shape
            `[B, L, N, C]`.
        """
        B, L, N = x.shape

        tetrahedra = x[:, :, self.flatten_tetrahedra]
        triangle = x[:, :, self.flatten_triangle]
        edge = x[:, :, self.flatten_edge]

        tetrahedra = tetrahedra.reshape(B * L, 1, len(self.flatten_tetrahedra))
        tetrahedra = self.conv_tetrahedra(tetrahedra)
        tetrahedra = tetrahedra.reshape(B, L, N)

        triangle = triangle.reshape(B * L, 1, len(self.flatten_triangle))
        triangle = self.conv_triangle(triangle)
        triangle = triangle.reshape(B, L, N)

        edge = edge.reshape(B * L, 1, len(self.flatten_edge))
        edge = self.conv_edge(edge)
        edge = edge.reshape(B, L, N)

        x = x + tetrahedra + triangle + edge

        x = self.readout(x)
        # x = x.view(B, N, self.prediction_seq_len)  # [B, L * N] -> [B, N, L]
        #
        # x = x.transpose(1, 2)
        #
        # x = self.rev_in(x, "denorm")
        return x


class Model(nn.Module):
    def __init__(self, configs: argparse.Namespace, network: dict):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(configs.c_out, 96, 1, batch_first=True)
        self.feature_fc = nn.Linear(96, configs.c_out)
        self.time_fc = nn.Linear(96, configs.pred_len)
        self.hcnn = HCNN(configs, network)
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
        x = self.hcnn(x)

        # apply fc to map to pred_len
        x = self.time_fc(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev_in(x, "denorm")
        return x
