from itertools import combinations
from typing import Dict

import networkx as nx
import numpy as np
from .tmfg import TMFG
from torch.utils.data import Dataset


class GraphGenerator:
    """
    Generate spatial and temporal TMFG for HNN with cross-correlation matrix.
    """

    def __init__(self, dataset: Dataset, N: int, L: int):
        self.dataset = dataset.data_x  # (L, N)
        self.N = N
        self.L = L

    def get_spatial_graph(self) -> Dict:
        """
        Construct amd Return the spatial TMFG for HNN, this graph only
        includes correlation between series, with size N.

        Returns:
            Dict: The spatial TMFG for HNN configs. It contains l1, l2, l3, l4,
            c1, c2, c3.
        """
        print("Generating spatial graph")
        corr = np.square(np.corrcoef(self.dataset, rowvar=False))
        _, _, adjacency_matrix = TMFG().fit_transform(
            weights=corr,
            output="unweighted_sparse_W_matrix",
        )
        assert len(adjacency_matrix) == self.N
        print(len(adjacency_matrix))
        return self._get_cliques(adjacency_matrix)

    def get_graph(self) -> Dict:
        """
        Construct amd return the full cross-correlation TMFG for HNN with size
         N * L.

        Returns:
            Dict: The correlation TMFG for HNN configs. It contains l1, l2, l3,
            l4, c1, c2, c3.
        """
        flatten_cross_correlation_matrix = self._get_flatten_cross_correlation_matrix(
            self.L
        )
        print("Starting TMFG")
        _, _, adjacency_matrix = TMFG().fit_transform(
            weights=flatten_cross_correlation_matrix,
            output="unweighted_sparse_W_matrix",
        )
        assert len(adjacency_matrix) == self.L * self.N
        print(len(adjacency_matrix))
        return self._construct_network(adjacency_matrix)

    def get_network(self) -> Dict:
        """
        Construct amd Return the cross-correlation TMFG for HNN. This returns an
        unprocessed structure, dictionary containing lists of cliques.

        Returns:
            Dict: The cross-correlation TMFG for HNN configs.
        """
        flatten_cross_correlation_matrix = self._get_flatten_cross_correlation_matrix(
            self.L
        ).astype(np.float32)
        _, _, adjacency_matrix = TMFG().fit_transform(
            weights=flatten_cross_correlation_matrix,
            output="unweighted_sparse_W_matrix",
        )
        assert len(adjacency_matrix) == self.L * self.N
        print(len(adjacency_matrix))
        network = self._get_cliques(adjacency_matrix)

        return network

    def get_adj_matrix(self) -> np.ndarray:
        flatten_cross_correlation_matrix = self._get_flatten_cross_correlation_matrix(
            self.L
        ).astype(np.float32)
        _, _, adjacency_matrix = TMFG().fit_transform(
            weights=flatten_cross_correlation_matrix,
            output="unweighted_sparse_W_matrix",
        )
        return adjacency_matrix

    def get_unfiltered_adj_matrix(self) -> np.ndarray:
        flatten_cross_correlation_matrix = self._get_flatten_cross_correlation_matrix(
            self.L
        ).astype(np.float32)
        return flatten_cross_correlation_matrix

    def _get_cross_correlation_matrix(self, lag: int) -> np.ndarray:
        """Calculate the cross-correlation matrix for the dataset.

        Args:
            lag (int): The lag for calculating the cross-correlation matrix.

        Returns:
            np.ndarray: The cross-correlation matrix with t0 and its lag. The
            shape is (N, N).
        """

        t0_data = self.dataset[:-lag].T
        tlag_data = self.dataset[lag:].T
        # Since np stack two arrays to calculate the correlation, resulting
        # shape of (2N, 2N). We only care about cross-correlation between t0
        # and tlag, so we only take the first N rows and last N columns (top
        # right N x N matrix).
        cross_correlation_matrix = np.corrcoef(t0_data, tlag_data)[: self.N, self.N:]
        return cross_correlation_matrix

    def _get_all_cross_correlation_matrix(self, max_lag: int) -> np.ndarray:
        """Calculate the cross-correlation matrix for the dataset.

        Args:
            max_lag (int): The maximum lag for calculating the cross-correlation matrix.

        Returns:
            np.ndarray: The cross-correlation matrix with t0 and its lag. The
            shape is (max_lag, N, N).
        """
        cross_correlation_matrix = np.zeros((max_lag, self.N, self.N))
        cross_correlation_matrix[0] = np.corrcoef(self.dataset.T)
        for lag in range(1, max_lag):
            cross_correlation_matrix[lag] = self._get_cross_correlation_matrix(lag)
            if lag % 10 == 0:
                print(f"Calculating cross-correlation matrix for lag {lag}")
        return cross_correlation_matrix

    def _get_flatten_cross_correlation_matrix(self, max_lag: int) -> np.ndarray:
        """Calculate the cross-correlation matrix for the dataset.

        Args:
            max_lag (int): The maximum lag for calculating the cross-correlation matrix.

        Returns:
            np.ndarray: The cross-correlation matrix with t0 and its lag. The
            shape is (max_lag * N, max_lag * N).
        """
        print("Starting flatten cross-correlation matrix")
        cross_correlation_matrices = self._get_all_cross_correlation_matrix(max_lag)
        flatten_cross_correlation_matrix = np.zeros(
            (max_lag * self.N, max_lag * self.N)
        )
        last_block_index = (max_lag - 1) * self.N
        for i in range(max_lag):
            row_start = (max_lag - i - 1) * self.N
            col_start = (max_lag - i - 1) * self.N
            flatten_cross_correlation_matrix[
                last_block_index:, col_start: col_start + self.N
            ] = cross_correlation_matrices[i]
            flatten_cross_correlation_matrix[
                row_start: row_start + self.N, last_block_index:
            ] = cross_correlation_matrices[i].T
            if i % 10 == 0:
                print(f"Flattening cross-correlation matrix for lag {i}")

        flatten_cross_correlation_matrix = np.square(flatten_cross_correlation_matrix)
        return flatten_cross_correlation_matrix

    def _get_cliques(self, adjacency_matrix):
        cliques = {1: [], 2: [], 3: [], 4: []}
        for clique in nx.enumerate_all_cliques(nx.from_numpy_array(adjacency_matrix)):
            cliques[len(clique)].append(clique)
        return cliques

    def _construct_network(self, adjacency_matrix):
        cliques = self._get_cliques(adjacency_matrix)
        for k, v in cliques.items():
            cliques[k] = [tuple(clique) for clique in v]
        connection1 = self._get_connection(cliques[1], cliques[2])
        connection2 = self._get_connection(cliques[2], cliques[3])
        connection3 = self._get_connection(cliques[3], cliques[4])

        params = {
            "l1": len(cliques[1]),
            "l2": len(cliques[2]),
            "l3": len(cliques[3]),
            "l4": len(cliques[4]),
            "c1": connection1,
            "c2": connection2,
            "c3": connection3,
        }
        return params

    def _get_connection(self, prev_cliques, next_cliques):
        connections = [[], []]
        if not prev_cliques or not next_cliques:
            return connections
        prev_cliques_index_map = {clique: i for i, clique in enumerate(prev_cliques)}
        prev_clique_size = len(prev_cliques[0])
        for j, next_clique in enumerate(next_cliques):
            for prev_clique in combinations(next_clique, prev_clique_size):
                i = prev_cliques_index_map.get(prev_clique)
                connections[0].append(i)
                connections[1].append(j)
        return connections
