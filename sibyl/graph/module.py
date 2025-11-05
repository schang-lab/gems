"""
Graph Neural Network Modules for Heterogeneous Graphs.
Available decoders:
    - BilinearDecoder: bilinear scoring of two node embeddings
    - MLPDecoder: MLP-based scoring of concatenated node embeddings
Available encoders:
    - FixedEmbedding: returns a fixed embedding of ones
    - LearnableProjection: linear projection from input features (e.g. LLM rep)
    - OneHotEncoder: one-hot encoding for categorical features
"""

import warnings
import collections
from typing import Dict, Tuple, Optional, List, Union, Literal, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GraphConv, SAGEConv, RGCNConv,
    HeteroConv,
)
from torch_geometric.data import HeteroData

from sibyl.constants.string_registry_survey import (
    INDIV_FEAT_ENCODING,
)


class CustomHeteroData(HeteroData):
    """
    Wrapper class for PyTorch Geometric's HeteroData with flexible validation.
    This class extends HeteroData to allow disabling or customizing sanity checks
    that are normally enforced by the base class.
    Inherits all functionality from torch_geometric.data.HeteroData.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run_sanity_check(self) -> None:
        return


class CustomModule(torch.nn.Module):
    """
    Base class for all custom neural network modules in the project.
    Provides utility methods for:
    - Printing trainable and total parameter counts
    - Computing gradient norms for gradient monitoring and clipping
    """
    def __init__(self):
        super().__init__()

    def print_trainable_params(self) -> None:
        """
        Print the total and trainable parameter counts for this module.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] "
              f"Total params: {total:,} | Trainable: {trainable:,}")

    @torch.no_grad()
    def get_grad_norm(self,
                      norm_type: Union[float, int, str] = 2,
                      only_trainable: bool = True) -> float:
        """
        Compute the global gradient norm across all parameters in this module.
        This method recursively processes all parameters and computes the p-norm of their gradients.
        Args:
            norm_type: Type of norm to compute. Can be:
                - A numeric p-value (e.g., 1, 2) for p-norm
                - 'inf' or float('inf') for infinity norm
            only_trainable: If True, only consider parameters with requires_grad=True.
                If False, compute norm across all parameters.
        Returns:
            The computed gradient norm as a float. Returns 0.0 if no gradients
            are present or if no parameters match the selection criteria.
        """
        params = (p for p in self.parameters()
                  if (p.requires_grad or not only_trainable))
        grads = []
        for p in params:
            g = p.grad
            if g is None:
                continue
            g = g.detach()
            if g.is_sparse:
                g = g.coalesce().values()
            grads.append(g)

        if not grads: return 0.0

        if norm_type in ("inf", float("inf")):
            max_vals = [g.abs().max().float() for g in grads]
            return torch.stack(max_vals).max().item()

        per_param = [torch.norm(g.float(), p=float(norm_type)) for g in grads]
        total = torch.norm(torch.stack(per_param), p=float(norm_type))
        return total.item()


@final
class BilinearDecoder(CustomModule):
    """
    Bilinear decoder for computing similarity scores between node embeddings.
    This module computes a bilinear product between two sets of node embeddings,
    commonly used for link prediction tasks in graph neural networks. The score
    for a pair (s, q) is computed as: s^T W q / tau, where W is a learnable
    weight matrix and tau is an optional temperature parameter.
    """
    def __init__(self,
                 embed_dims: Tuple[int, int],
                 bias: bool = False,
                 is_identity: bool = False,
                 learnable_temp: bool = False) -> None:
        """
        Args:
            embed_dims: Tuple of (source_dim, query_dim) for input embedding dimensions.
            bias: Whether to include bias term in bilinear layer.
            is_identity: If True, initialize bilinear weights to identity matrix
                (requires embed_dims[0] == embed_dims[1]) and freeze them.
            learnable_temp: If True, use a learnable temperature parameter
                to scale the output scores.
        """
        super().__init__()
        self.bilinear = torch.nn.Bilinear(
            embed_dims[0], embed_dims[1], 1, bias=bias
        )
        if is_identity:
            assert embed_dims[0] == embed_dims[1]
            with torch.no_grad():
                self.bilinear.weight.zero_()
                torch.nn.init.eye_(self.bilinear.weight[0])
            self.bilinear.weight.requires_grad_(False)
        else:
            torch.nn.init.xavier_uniform_(self.bilinear.weight)
        if bias:
            torch.nn.init.zeros_(self.bilinear.bias)
        self.learnable_temp = learnable_temp
        self.tau = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32),
            requires_grad=self.learnable_temp
        )
        self.print_trainable_params()

    @classmethod
    def from_dict(cls, state_dict: collections.OrderedDict):
        """
        Load BilinearDecoder from a PyTorch state_dict.
        """
        w_shape = state_dict['bilinear.weight'].shape
        embed_dims = (w_shape[1], w_shape[2])
        decoder = cls(
            embed_dims=embed_dims,
            bias=("bilinear.bias" in state_dict),
        )
        decoder.load_state_dict(state_dict, strict=False)
        return decoder

    def forward(self, h_s: torch.Tensor, h_q: torch.Tensor) -> torch.Tensor:
        """
        Compute element-wise bilinear scores for paired embeddings.
        Args:
            h_s: Source embeddings of shape (batch_size, source_dim).
            h_q: Query embeddings of shape (batch_size, query_dim).
        Returns:
            Scores of shape (batch_size,) representing similarity between
            corresponding pairs.
        """
        return self.bilinear(h_s, h_q).squeeze(-1) / self.tau

    @torch.no_grad()
    def _choose_left(self, M: int, D1: int, D2: int, N: int) -> bool:
        """Choose optimal matrix multiplication order for computational efficiency."""
        left = M * D2 * (D1 + N)  # cost of left GEMM
        right = N * D1 * (D2 + M)  # cost of right GEMM
        return left < right

    def pairwise(self, h_s: torch.Tensor, h_q: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise bilinear scores between all combinations of embeddings.
        This method computes scores for all pairs (h_s[i], h_q[j]) efficiently
        using optimized matrix multiplication order. Different from forward()
        which computes only paired scores.
        Args:
            h_s: Source embeddings of shape (M, source_dim).
            h_q: Query embeddings of shape (N, query_dim).
        Returns:
            Scores of shape (M, N) where scores[i, j] represents the similarity
            between h_s[i] and h_q[j].
        """
        assert h_s.dim() == 2 and h_q.dim() == 2
        M, D1 = h_s.shape
        N, D2 = h_q.shape
        W = self.bilinear.weight.squeeze(0)
        do_left = self._choose_left(M, D1, D2, N)
        if do_left:
            scores = (h_s @ W) @ h_q.t()
        else:
            scores = h_s @ (W @ h_q.t())
        if self.bilinear.bias is not None:
            scores = scores + self.bilinear.bias
        return scores / self.tau


class MLP(CustomModule):
    """
    Multi-Layer Perceptron with configurable architecture.
    A flexible MLP implementation with support for:
    - Multiple hidden layers with configurable dimensions
    - Customizable activation functions
    - Dropout regularization
    Possible use cases in the project:
    - Converting LLM embeddings to graph node embeddings
    - Concatenating GNN outputs for link prediction scoring (instead of bilineardecoder)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 out_dim: int,
                 nonlin_fn: str,
                 p_dropout: float,
                 n_layers: Optional[int] = 1) -> None:
        """
        Args:
            input_dim: Dimension of input features.
            hidden_dims: List of hidden layer dimensions. Length must be n_layers - 1.
            out_dim: Dimension of output features.
            nonlin_fn: Name of activation function (e.g., 'relu', 'tanh', 'elu').
                Must be available in torch.nn.functional.
            p_dropout: Dropout probability applied after each hidden layer.
                Same dropout rate across all layers.
            n_layers: Total number of layers (including output layer). Must be >= 1.
        """
        super().__init__()
        assert n_layers >= 1, "--> MLP: n_layers must be at least 1."
        assert len(hidden_dims) == n_layers - 1, (
            "--> MLP: len(hidden_dims) must be equal to n_layers - 1."
        )
        self.input_dim, self.hidden_dims, self.out_dim = (
            input_dim, hidden_dims, out_dim
        )
        self.n_layers = n_layers
        self.nonlin_fn = nonlin_fn
        self.p_dropout = p_dropout

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, out_dim))
        self.layers = torch.nn.ModuleList(layers)
        self.print_trainable_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.Linear):
                x = getattr(F, self.nonlin_fn)(layer(x))
                x = F.dropout(x, p=self.p_dropout, training=self.training)
        return self.layers[-1](x)


@final
class MLPDecoder(MLP):
    """
    Wrapper for MLP to be used as a decoder.
    Gets two node embeddings and outputs a score for the pair.
    """
    def __init__(self,
                 embed_dims: Tuple[int, int],
                 mlp_configs: Dict) -> None:
        super().__init__(
            input_dim=embed_dims[0]+embed_dims[1],
            hidden_dims=mlp_configs['mlp_hidden_dims'],
            out_dim=1,
            nonlin_fn=mlp_configs['mlp_nonlin_fn'],
            p_dropout=mlp_configs['mlp_p_dropout'],
            n_layers=mlp_configs['mlp_n_layers'],
        )

    def forward(self, h_s: torch.Tensor, h_q: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat([h_s, h_q], dim=-1)).squeeze(-1)

        
@final
class FixedEmbedding(CustomModule):
    """
    Fixed (non-learnable) embedding layer that returns ones.
    This module always returns a tensor of ones regardless of input,
    useful as a baseline or placeholder embedding that doesn't learn from data.
    """
    def __init__(self):
        super().__init__()
        self.print_trainable_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a fixed embedding of ones for each input.
        Args:
            x: Input tensor of shape (batch_size, *).
        Returns:
            Tensor of ones with shape (batch_size, 1).
        """
        assert x.dim() == 2, "Input must be a 2D tensor."
        n_batch = x.shape[0]
        return torch.ones(n_batch, 1, dtype=torch.float32,
                          requires_grad=False, device=x.device)


@final
class LearnableProjection(CustomModule):
    """
    Learnable linear projection with optional input offset and dropout.
    Projects input features (e.g. LLM representation) through a linear layer
    with optional centering by subtracting an offset before projection.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 p_dropout: float,
                 input_offset: Optional[torch.Tensor] = None,
                 use_bias: bool = True):
        """
        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            p_dropout: Dropout probability applied 'after' projection.
            input_offset: Optional offset to subtract from input before projection.
                If None, uses zero offset. Should be 1D tensor of size input_dim.
            use_bias: Whether to include bias in the linear layer.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        if input_offset is None:
            offset = torch.zeros(1, input_dim)
        else:
            assert input_offset.ndim == 1 and input_offset.shape[0] == input_dim
            offset = input_offset.unsqueeze(0)
        self.register_buffer("input_offset", offset)
        self.p_dropout = p_dropout
        self.print_trainable_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            self.linear(x - self.input_offset),
            p=self.p_dropout, training=self.training
        )


class OneHotEncoder(CustomModule):
    """
    A lookup table that takes node index and returns one-hot emeddings.
    """
    def __init__(self,
                 embedding_dim: int,
                 input_dim: int = 1,
                 n_attr: int = 1,
                 set_trainable: bool = True,
                 print_params: bool = True):
        """
        Args:
            embedding_dim: Output embedding dimension to be used as graph input feature.
            input_dim: vocabulary size; number of nodes to be encoded.
                when n_attr = 1, the learnable table has size (input_dim, embedding_dim)
                and during forward pass, given node index i, returns the i-th row of table.
            n_attr: this flag only used for OneHotAttributeEncoder child class
                refer to docstring of OneHotAttributeEncoder for details.
        """
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(input_dim, embedding_dim) for _ in range(n_attr)
        ])
        for _module in self.emb_layers:
            _module.weight.requires_grad = set_trainable
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.n_attr = n_attr
        if print_params:
            self.print_trainable_params()

    def forward(self,
                x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of one-hot encoding, replaces missing values with zero vectors.
        When n_attr > 1, concatenate embeddings.
        """
        assert x.dim() == 2, "Input must be a 2D tensor."
        assert x.dtype in (torch.long, torch.int)
        x = x[:, x.shape[1] - self.n_attr :]
        embeds = []
        for i, emb_layer in enumerate(self.emb_layers):
            xi = x[:, i]
            mask = (xi == -1)
            xi_safe = xi.clone()
            xi_safe[mask] = 0
            e = emb_layer(xi_safe)
            if mask.any():
                e[mask] = 0.0
            embeds.append(e)
        return torch.cat(embeds, dim=1)


@final
class OneHotAttributeEncoder(OneHotEncoder):
    """
    One-hot encoder for individual feature.
    This class used for ablation only
    where individual nodes' features are defined by the individual's individual features.
    """
    def __init__(self,
                 embedding_dim: int, source: str,
                 attr_list: List[str]):
        """
        Args:
            attr_list: example is ['age', 'gender']
                then n_attr = 2, and for each attribute search in INDIV_FEAT_ENCODING
                to get the number of categories for that attribute.
                self.emb_layers will have two embedding layers,
                one for age and another for gender, each with size (num_categories, embedding_dim)
        """
        super().__init__(
            embedding_dim=embedding_dim,
            n_attr=len(attr_list),
            print_params=False,
        )
        self.emb_layers = nn.ModuleList([
            nn.Embedding(
                len(INDIV_FEAT_ENCODING[source][attr]), embedding_dim
            ) for attr in attr_list
        ])
        self.print_trainable_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x[:,:-1])


class GeneralHetero(torch.nn.Module):
    """
    Base class for heterogeneous graph neural networks.
    This abstract base class provides common functionality for heterogeneous GNNs
    that support multiple node types and edge types. It handles:
    - Flexible configuration per node/edge type or globally
    - Layer normalization and residual connections
    - Type checking and validation during initialization
    - A dry run to infer dimensions and validate architecture
    Subclasses implement specific GNN architectures (GraphConv, GAT, SAGE, RGCN).
    """
    def __init__(self,
                 graph_for_dry_run: CustomHeteroData,
                 hidden_dims: Union[List[int], Dict[str, List[int]]],
                 out_dim: Union[int, Dict[str, int]],
                 num_layers: int,
                 dropout: float,
                 nonlin_fn: Union[str, Dict[str, str]] = 'relu',
                 aggr_method: Union[
                     str, Dict[Tuple[str, str, str], str]
                 ] = 'mean',
                 hetero_aggr_method: str = 'mean',
                 norm_type: Literal['layer', 'none'] = 'none',
                 ln_eps: float = 1e-5,
                 ln_affine: bool = True,
                 use_residual: bool = False,
                 **kwargs,
    ):
        """
        The module performs a dry run with the provided graph to infer node and
        edge types, validate configurations, and set up dimensions.
        Args:
            graph_for_dry_run: Sample HeteroData graph used to infer node/edge types
                and dimensions. Should contain representative data.
            hidden_dims: Hidden dimensions for each layer. Can be:
                - List[int]: Same dimensions for all node types
                - Dict[str, List[int]]: Per-node-type dimensions
            out_dim: Output dimension(s). Can be:
                - int: Same for all node types
                - Dict[str, int]: Per-node-type output dimensions
            num_layers: Number of GNN layers.
            dropout: Dropout probability applied after each layer.
            nonlin_fn: Activation function name (e.g., 'relu', 'elu'). Can be:
                - str: Same for all node types
                - Dict[str, str]: Per-node-type activation
            aggr_method: Aggregation for messages within each edge type (e.g., 'mean',
                'sum', 'max'). Can be:
                - str: Same for all edge types
                - Dict[Tuple[str, str, str], str]: Per-edge-type aggregation
            hetero_aggr_method: Aggregation across different edge types for each node
                ('mean', 'sum', or 'maskedmean').
            norm_type: Normalization type ('layer' or 'none').
            ln_eps: LayerNorm epsilon (only used if norm_type='layer').
            ln_affine: Whether LayerNorm has learnable parameters.
            use_residual: Whether to use residual connections.
            **kwargs: Additional arguments for subclasses.
        """
        assert isinstance(graph_for_dry_run, CustomHeteroData)
        super().__init__()
        # infer node and edge types from dryrun
        self.node_types: set[str] = set(graph_for_dry_run.node_types)
        self.edge_types: set[Tuple[str, str, str]] = (
            graph_for_dry_run.edge_types
        )
        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            if (dst, rel, src) not in self.edge_types:
                warnings.warn(f"{self.__class__.__name__}: "
                              f"Edge {edge_type} not bidirectional")
        # typecheck
        for node_config in [hidden_dims, out_dim, nonlin_fn]:
            if isinstance(node_config, dict):
                assert self.node_types <= set(node_config.keys()), (
                    f"Node config {set(node_config.keys())} mismatch "
                    f"{self.node_types}"
                )                    
        for edge_config in [aggr_method]:
            if isinstance(edge_config, dict):
                assert self.edge_types <= set(edge_config.keys()), (
                    f"Edge config {set(edge_config.keys())} mismatch "
                    f"{self.edge_types}"
                )
        # sanitycheck
        if isinstance(hidden_dims, list):
            assert len(hidden_dims) == num_layers, (
                f"hidden_dims {len(hidden_dims)} mismatch n = {num_layers}"
            )
        elif isinstance(hidden_dims, dict):
            assert all(
                len(hidden_dims[node_type]) == num_layers
                for node_type in self.node_types
            ), f"hidden_dims {hidden_dims} mismatch n = {num_layers}"
        # store parameters
        self.num_layers: int = num_layers
        self.dropout: float = dropout
        self.hetero_aggr_method: str = hetero_aggr_method
        self.nonlin_fn: Dict[str, str] = (
            nonlin_fn if isinstance(nonlin_fn, dict) else
            {nt: nonlin_fn for nt in self.node_types}
        )
        self.aggr_method: Dict[Tuple[str, str, str], str] = (
            aggr_method if isinstance(aggr_method, dict) else
            {et: aggr_method for et in self.edge_types}
        )
        self.hidden_dims: Dict[str, List[int]] = (
            hidden_dims if isinstance(hidden_dims, dict) else
            {nt: hidden_dims for nt in self.node_types}
        )
        self.out_dim: Dict[str, int] = (
            out_dim if isinstance(out_dim, dict) else
            {nt: out_dim for nt in self.node_types}
        )
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        # normalization layer
        assert norm_type in ['layer', 'none']
        self.norm_type = norm_type
        self.norms: Optional[torch.nn.ModuleList] = None
        if self.norm_type == 'layer':
            self.norms = nn.ModuleList([
                nn.ModuleDict({
                    nt: nn.LayerNorm(self.hidden_dims[nt][layer_idx],
                                     eps=ln_eps,
                                     elementwise_affine=ln_affine)
                    for nt in self.node_types
                }) for layer_idx in range(self.num_layers)
            ])
        # residual connections
        self.use_residual: bool = use_residual
        if self.use_residual:
            in_dims_by_layer = []
            in_dims_by_layer.append(
                {nt: graph_for_dry_run.x_dict[nt].size(-1)
                 for nt in self.node_types}
            )
            for l in range(1, self.num_layers):
                in_dims_by_layer.append(
                    {nt: self.hidden_dims[nt][l-1]
                     for nt in self.node_types}
                )
            self.res_proj = nn.ModuleList([
                nn.ModuleDict({
                    nt: (nn.Identity()
                        if in_dims_by_layer[layer_idx][nt] == self.hidden_dims[nt][layer_idx]
                        else nn.Linear(
                            in_dims_by_layer[layer_idx][nt],
                            self.hidden_dims[nt][layer_idx],
                            bias=False
                        )) for nt in self.node_types
                }) for layer_idx in range(self.num_layers)
            ])
        return
        
    def print_trainable_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] "
              f"Total params: {total:,} | Trainable: {trainable:,}")


@final
class GeneralHeteroGraphConv(GeneralHetero):

    def __init__(self,
                 graph_for_dry_run: CustomHeteroData,
                 hidden_dims: Union[List[int], Dict[str, List[int]]],
                 out_dim: Union[int, Dict[str, int]], 
                 num_layers: int,
                 dropout: float,
                 nonlin_fn: Union[str, Dict[str, str]] = 'relu',
                 aggr_method: Union[
                     str, Dict[Tuple[str, str, str], str]
                 ] = 'mean',
                 hetero_aggr_method: str = 'mean',
                 norm_type: Literal['layer', 'none'] = 'none',
                 ln_eps: float = 1e-5,
                 ln_affine: bool = True,
                 use_residual: bool = False,
                 **kwargs,
    ) -> None:
        """
        GraphConv implementation of GeneralHetero
        (HeteroGNN with flexible node and edge types).
        """
        super().__init__(graph_for_dry_run=graph_for_dry_run,
                         hidden_dims=hidden_dims,
                         out_dim=out_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         nonlin_fn=nonlin_fn,
                         aggr_method=aggr_method,
                         norm_type=norm_type,
                         ln_eps=ln_eps,
                         ln_affine=ln_affine,
                         hetero_aggr_method=hetero_aggr_method,
                         use_residual=use_residual,
                         **kwargs)
        
        in_channels: Dict[str, int] = {nt: -1 for nt in self.node_types}
        for layer_idx in range(self.num_layers):
            convs = {
                et: GraphConv(
                    in_channels=(in_channels[et[0]], in_channels[et[2]]),
                    out_channels=self.hidden_dims[et[2]][layer_idx],
                    aggr=self.aggr_method[et],
                ) for et in self.edge_types
            }
            self.layers.append(
                HeteroConv(convs, aggr=self.hetero_aggr_method
                           if self.hetero_aggr_method in ['mean', 'sum']
                           else None
                )
            )
            in_channels = {
                nt: self.hidden_dims[nt][layer_idx]
                for nt in self.node_types
            }
        self.lin: nn.ModuleDict = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], self.out_dim[nt])
            for nt in self.node_types
        })
        self.eval()
        with torch.no_grad():
            _ = self.forward(
                x_dict=graph_for_dry_run.x_dict,
                edge_index_dict=graph_for_dry_run.edge_index_dict,
            )
        self.train()
        self.print_trainable_params()
        return

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.LongTensor],
                edge_weight_dict: Optional[
                    Dict[Tuple[str, str, str], torch.Tensor]
                ] = None,
                **kwargs,
                ) -> Dict[str, torch.Tensor]:
        """
        Currently supports node updates every layer.
        If edge updates (e.g. GraphSAGE) are needed, need modification.
        """
        for layer_idx, conv in enumerate(self.layers):
            x_in = x_dict
            if edge_weight_dict is not None:
                x_dict = conv(x_dict=x_dict,
                              edge_index_dict=edge_index_dict,
                              edge_weight_dict=edge_weight_dict)
            else:
                x_dict = conv(x_dict=x_dict,
                              edge_index_dict=edge_index_dict)                              
            if 'debug' in kwargs and 'pdb_forward_path' in kwargs['debug']:
                import pdb; pdb.set_trace()
            if self.hetero_aggr_method not in ['mean', 'sum']:
                if self.hetero_aggr_method == "maskedmean":
                    for nt in x_dict.keys():
                        mask = x_dict[nt].ne(0).any(dim=-1)
                        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        x_dict[nt] = x_dict[nt].sum(dim=1) / counts
                else: # any non-standard aggregation should be handled here
                    raise ValueError(f"Invalid aggr {self.hetero_aggr_method}")
            if self.use_residual:
                x_dict = {
                    nt: x_dict[nt] + self.res_proj[layer_idx][nt](x_in[nt])
                    for nt in x_dict.keys()
                }
            if self.norm_type != 'none':
                x_dict = {
                    nt: self.norms[layer_idx][nt](x)
                    for nt, x in x_dict.items()
                }
            x_dict = {
                nt: F.dropout(
                    getattr(F, self.nonlin_fn[nt])(x),
                    p=self.dropout, training=self.training
                ) for nt, x in x_dict.items()
            }
        return {nt: self.lin[nt](x) for nt, x in x_dict.items()}


@final
class GeneralHeteroGAT(GeneralHetero):
    def __init__(self,
                 graph_for_dry_run: CustomHeteroData,
                 hidden_dims: Union[List[int], Dict[str, List[int]]],
                 out_dim: Union[int, Dict[str, int]],
                 num_layers: int,
                 dropout: float,
                 heads: List[int],
                 concat: bool,
                 attn_dropout: float,
                 negative_slope: float,
                 nonlin_fn: Union[str, Dict[str, str]] = 'relu',
                 aggr_method: Union[
                     str, Dict[Tuple[str, str, str], str]
                 ] = 'mean',
                 hetero_aggr_method: str = 'mean',
                 norm_type: Literal['layer', 'none'] = 'none',
                 ln_eps: float = 1e-5,
                 ln_affine: bool = True,
                 use_residual: bool = False,
                 **kwargs) -> None:
        """
        GATConv implementation of GeneralHetero
        (HeteroGNN with flexible node / edge types)
        """
        super().__init__(graph_for_dry_run=graph_for_dry_run,
                         hidden_dims=hidden_dims,
                         out_dim=out_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         nonlin_fn=nonlin_fn,
                         aggr_method=aggr_method,
                         hetero_aggr_method=hetero_aggr_method,
                         norm_type=norm_type,
                         ln_eps=ln_eps,
                         ln_affine=ln_affine,
                         use_residual=use_residual,
                         **kwargs)
        
        # GAT-specific parameters
        self.heads : List[int] = heads
        self.concat : bool = concat
        self.attn_dropout : float = attn_dropout
        self.negative_slope : float = negative_slope

        in_channels: Dict[str, int] = {nt: -1 for nt in self.node_types}
        for layer_idx in range(self.num_layers):
            convs = {}
            for et in self.edge_types:
                src, _, dst = et
                hidden_dim = self.hidden_dims[dst][layer_idx]
                current_layer_heads = self.heads[layer_idx]
                out_per_head = hidden_dim
                if self.concat:
                    assert hidden_dim % current_layer_heads == 0, (
                        f"Hidden dim {hidden_dim} must be divisible by "
                        f"heads {current_layer_heads} when concat=True"
                    )
                    out_per_head = hidden_dim // current_layer_heads
                convs[et] = GATConv(
                    in_channels=(in_channels[src], in_channels[dst]),
                    out_channels=out_per_head,
                    heads=current_layer_heads,
                    concat=self.concat,
                    dropout=self.attn_dropout,
                    add_self_loops=False,
                    aggr=self.aggr_method[et],
                    negative_slope=self.negative_slope,
                )
            self.layers.append(
                HeteroConv(convs, aggr=self.hetero_aggr_method
                           if self.hetero_aggr_method in ['mean', 'sum']
                           else None
                )
            )
            in_channels = {
                nt: self.hidden_dims[nt][layer_idx]
                for nt in self.node_types
            }
        self.lin = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], self.out_dim[nt])
            for nt in self.node_types
        })
        self.eval()
        with torch.no_grad():
            _ = self.forward(
                x_dict=graph_for_dry_run.x_dict,
                edge_index_dict=graph_for_dry_run.edge_index_dict,
            )
        self.train()
        self.print_trainable_params()

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.LongTensor],
                edge_attr_dict: Optional[
                    Dict[Tuple[str, str, str], torch.Tensor]
                ] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        for layer_idx, conv in enumerate(self.layers):
            x_in = x_dict
            if edge_attr_dict is not None:
                x_dict = conv(x_dict=x_dict,
                              edge_index_dict=edge_index_dict,
                              edge_attr_dict=edge_attr_dict)
            else:
                x_dict = conv(x_dict=x_dict,
                              edge_index_dict=edge_index_dict)
            if 'debug' in kwargs and 'pdb_forward_path' in kwargs['debug']:
                import pdb; pdb.set_trace()
            if self.hetero_aggr_method not in ['mean', 'sum']:
                if self.hetero_aggr_method == "maskedmean":
                    for nt in x_dict.keys():
                        mask = x_dict[nt].ne(0).any(dim=-1)
                        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        x_dict[nt] = x_dict[nt].sum(dim=1) / counts
                else: # any non-standard aggregation should be handled here
                    raise ValueError(f"Invalid aggr {self.hetero_aggr_method}")
            if self.use_residual:
                x_dict = {
                    nt: x_dict[nt] + self.res_proj[layer_idx][nt](x_in[nt])
                    for nt in x_dict.keys()
                }
            if self.norm_type != 'none':
                x_dict = {
                    nt: self.norms[layer_idx][nt](x)
                    for nt, x in x_dict.items()
                }
            x_dict = {
                nt: F.dropout(
                    getattr(F, self.nonlin_fn[nt])(h),
                    p=self.dropout, training=self.training
                ) for nt, h in x_dict.items()
            }
        return {nt: self.lin[nt](x) for nt, x in x_dict.items()}


@final
class GeneralHeteroSAGE(GeneralHetero):
    def __init__(self,
                 graph_for_dry_run: CustomHeteroData,
                 hidden_dims: Union[List[int], Dict[str, List[int]]],
                 out_dim: Union[int, Dict[str, int]],
                 num_layers: int,
                 dropout: float,
                 nonlin_fn: Union[str, Dict[str, str]] = 'relu',
                 aggr_method: Union[str, Dict[Tuple[str, str, str], str]] = 'mean',
                 hetero_aggr_method: str = 'mean',
                 norm_type: Literal['layer', 'none'] = 'none',
                 ln_eps: float = 1e-5,
                 ln_affine: bool = True,
                 use_residual: bool = False,
                 # SAGE-specific knobs:
                 sage_normalize: bool = True,   # L2-normalize output embeddings per layer
                 sage_root_weight: bool = True,  # include center node transform
                 sage_project: bool = False,     # extra projection in SAGEConv
                 **kwargs) -> None:
        """
        GraphSAGE implementation (heterogeneous).
        - When set use_residual=True, consider sage_root_weight=False to avoid double skip paths.
        """
        super().__init__(graph_for_dry_run=graph_for_dry_run,
                         hidden_dims=hidden_dims,
                         out_dim=out_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         nonlin_fn=nonlin_fn,
                         aggr_method=aggr_method,
                         hetero_aggr_method=hetero_aggr_method,
                         norm_type=norm_type,
                         ln_eps=ln_eps,
                         ln_affine=ln_affine,
                         use_residual=use_residual,
                         **kwargs)

        self.sage_normalize: bool = sage_normalize
        self.sage_root_weight: bool = sage_root_weight
        self.sage_project: bool = sage_project

        in_channels: Dict[str, int] = {nt: -1 for nt in self.node_types}
        for layer_idx in range(self.num_layers):
            convs: Dict[Tuple[str, str, str], nn.Module] = {}
            for et in self.edge_types:
                src, _, dst = et
                convs[et] = SAGEConv(
                    in_channels=(in_channels[src], in_channels[dst]),
                    out_channels=self.hidden_dims[dst][layer_idx],
                    aggr=self.aggr_method[et],          # 'mean'/'max'/'add'
                    normalize=self.sage_normalize,      # L2 normalize outputs
                    root_weight=self.sage_root_weight,  # include center transform
                    project=self.sage_project,          # final projection
                    bias=True,
                )
            self.layers.append(
                HeteroConv(convs, aggr=self.hetero_aggr_method
                           if self.hetero_aggr_method in ['mean', 'sum']
                           else None
                )
            )
            in_channels = {
                nt: self.hidden_dims[nt][layer_idx]
                for nt in self.node_types
            }
        self.lin = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], self.out_dim[nt])
            for nt in self.node_types
        })
        self.eval()
        with torch.no_grad():
            _ = self.forward(
                x_dict=graph_for_dry_run.x_dict,
                edge_index_dict=graph_for_dry_run.edge_index_dict,
            )
        self.train()
        self.print_trainable_params()

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.LongTensor],
                **kwargs) -> Dict[str, torch.Tensor]:

        for layer_idx, conv in enumerate(self.layers):
            x_in = x_dict
            x_dict = conv(x_dict=x_dict,
                          edge_index_dict=edge_index_dict)
            if 'debug' in kwargs and 'pdb_forward_path' in kwargs['debug']:
                import pdb; pdb.set_trace()
            if self.hetero_aggr_method not in ['mean', 'sum']:
                if self.hetero_aggr_method == "maskedmean":
                    for nt in x_dict.keys():
                        mask = x_dict[nt].ne(0).any(dim=-1)
                        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        x_dict[nt] = x_dict[nt].sum(dim=1) / counts
                else: # any non-standard aggregation should be handled here
                    raise ValueError(f"Invalid aggr {self.hetero_aggr_method}")
            if self.use_residual:
                x_dict = {
                    nt: x_dict[nt] + self.res_proj[layer_idx][nt](x_in[nt])
                    for nt in x_dict.keys()
                }
            if self.norm_type != 'none':
                x_dict = {
                    nt: self.norms[layer_idx][nt](h)
                    for nt, h in x_dict.items()
                }
            x_dict = {
                nt: F.dropout(
                    getattr(F, self.nonlin_fn[nt])(h),
                    p=self.dropout, training=self.training
                ) for nt, h in x_dict.items()
            }

        return {nt: self.lin[nt](h) for nt, h in x_dict.items()}


@final
class GeneralHeteroRGCNConv(GeneralHetero):

    def __init__(self,
                 graph_for_dry_run: CustomHeteroData,
                 hidden_dims: Union[List[int], Dict[str, List[int]]],
                 out_dim: Union[int, Dict[str, int]], 
                 num_layers: int,
                 dropout: float,
                 nonlin_fn: Union[str, Dict[str, str]] = 'relu',
                 aggr_method: Union[
                     str, Dict[Tuple[str, str, str], str]
                 ] = 'mean',
                 hetero_aggr_method: str = 'mean',
                 norm_type: Literal['layer', 'none'] = 'none',
                 ln_eps: float = 1e-5,
                 ln_affine: bool = True,
                 use_residual: bool = False,
                 **kwargs,
    ) -> None:
        """
        RGCNConv implementation of GeneralHetero
        (HeteroGNN with flexible node and edge types).
        """
        super().__init__(graph_for_dry_run=graph_for_dry_run,
                         hidden_dims=hidden_dims,
                         out_dim=out_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         nonlin_fn=nonlin_fn,
                         aggr_method=aggr_method,
                         norm_type=norm_type,
                         ln_eps=ln_eps,
                         ln_affine=ln_affine,
                         hetero_aggr_method=hetero_aggr_method,
                         use_residual=use_residual,
                         **kwargs)

        # For RGCNConv we need concrete input sizes (no lazy -1).
        in_channels: Dict[str, int] = {
            nt: int(graph_for_dry_run[nt].x.size(-1))
            for nt in self.node_types
        }
        for layer_idx in range(self.num_layers):
            convs = {
                et: RGCNConv(
                    in_channels=(in_channels[et[0]], in_channels[et[2]]),
                    out_channels=self.hidden_dims[et[2]][layer_idx],
                    num_relations=1,
                    aggr=self.aggr_method[et],
                    root_weight=True,
                ) for et in self.edge_types
            }
            self.layers.append(
                HeteroConv(
                    convs,
                    aggr=self.hetero_aggr_method
                    if self.hetero_aggr_method in ['mean', 'sum'] else None
                )
            )
            in_channels = {
                nt: self.hidden_dims[nt][layer_idx]
                for nt in self.node_types
            }
        self.lin: nn.ModuleDict = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], self.out_dim[nt])
            for nt in self.node_types
        })
        # Warn once if I accidently try to feed edge weights (RGCNConv ignores):
        self._warned_edge_weight = False
        self.eval()
        with torch.no_grad():
            _ = self.forward(
                x_dict=graph_for_dry_run.x_dict,
                edge_index_dict=graph_for_dry_run.edge_index_dict,
            )
        self.train()
        self.print_trainable_params()
        return

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.LongTensor],
                edge_weight_dict: Optional[
                    Dict[Tuple[str, str, str], torch.Tensor]
                ] = None,
                **kwargs,
                ) -> Dict[str, torch.Tensor]:
        # RGCNConv doesn't take edge_weight; keep signature but ignore politely.
        if edge_weight_dict is not None and not self._warned_edge_weight:
            warnings.warn(
                "GeneralHeteroRGCNConv ignores `edge_weight_dict` because "
                "`RGCNConv.forward()` does not accept `edge_weight`. "
                "If you need weighted relations, consider subclassing RGCNConv "
                "to add `edge_weight` to `message()` per PyG guidance.",
                stacklevel=2
            )
            self._warned_edge_weight = True

        for layer_idx, conv in enumerate(self.layers):
            x_in = x_dict
            # One relation per edge type -> all-zeros edge_type for each subgraph:
            edge_type_dict = {
                et: edge_index_dict[et].new_zeros(
                        (edge_index_dict[et].size(1),), dtype=torch.long)
                for et in edge_index_dict
            }
            x_dict = conv(x_dict=x_dict,
                          edge_index_dict=edge_index_dict,
                          edge_type_dict=edge_type_dict)
            if 'debug' in kwargs and 'pdb_forward_path' in kwargs['debug']:
                import pdb; pdb.set_trace()
            if self.hetero_aggr_method not in ['mean', 'sum']:
                if self.hetero_aggr_method == "maskedmean":
                    for nt in x_dict.keys():
                        mask = x_dict[nt].ne(0).any(dim=-1)
                        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        x_dict[nt] = x_dict[nt].sum(dim=1) / counts
                else:
                    raise ValueError(f"Invalid aggr {self.hetero_aggr_method}")
            if self.use_residual:
                x_dict = {
                    nt: x_dict[nt] + self.res_proj[layer_idx][nt](x_in[nt])
                    for nt in x_dict.keys()
                }
            if self.norm_type != 'none':
                x_dict = {
                    nt: self.norms[layer_idx][nt](x)
                    for nt, x in x_dict.items()
                }
            x_dict = {
                nt: F.dropout(
                    getattr(F, self.nonlin_fn[nt])(x),
                    p=self.dropout, training=self.training
                ) for nt, x in x_dict.items()
            }

        return {nt: self.lin[nt](x) for nt, x in x_dict.items()}