import random
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import chain

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from sibyl.graph.module import CustomHeteroData
from sibyl.constants.string_registry_graph import (
    ALLOWED_EDGE_TYPES,
    ALLOWED_NODE_TYPES,
)
from sibyl.utils.misc_utils import StringIndexer, PairIndexer


class GraphInfo:
    
    def __init__(self, nodes: List[Dict]):
        """
        self.node_types: set[str] ex. {'indiv', 'question'}
        self.node_labels: Dict[str, Indexer]
            ex. node_labels['indiv'] : supports
                .get_index('1071748') -> 0
                .get_string(0) -> '1071748'
        self.edge_labels: Dict[Tuple[str, str, str], PairIndexer]
            ex. edge_labels[('indiv', 'responds', 'question')] supports
                .get_index(('1071748', 'q1')) -> (1234, 1.0)
                .get_pair(1234) -> ('1071748', 'q1', 1.0)
        """
        self.node_types: set[str] = set()
        self.node_labels: Dict[str, StringIndexer] = {}
        self.edge_labels: Dict[Tuple[str, str, str], PairIndexer] = {}
        
        node_dict : Dict[str, List[str]] = {}
        edge_dict : Dict[str, List[Tuple[str, str, float]]] = {}

        for node in tqdm(nodes, desc="Building GraphInfo"):
            # update node types
            node_type = node['node_type']
            assert node_type in ALLOWED_NODE_TYPES
            self.node_types.add(node_type)
            # maintain list of node labels
            label = node['label']
            node_dict.setdefault(node_type, []).append(label)
            # maintain edge information
            edges = node.get('edges', {})
            for edge_type, edge_list in edges.items():
                assert edge_type in ALLOWED_EDGE_TYPES
                if edge_type not in edge_dict:
                    edge_dict[edge_type] = []
                edge_dict[edge_type].extend([
                    (label, to, weight)
                    for to, weight in edge_list
                ])

        for node_type in node_dict:
            self.node_labels[node_type] = StringIndexer(
                node_dict[node_type]
            )
        for edge_type in edge_dict:
            self.edge_labels[edge_type] = PairIndexer(
                edge_dict[edge_type]
            )
        print("--> GraphInfo is initialized.")
        return
    
    def get_labels(self,
                   node_type: str
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get all labels of nodes of type node_type.
        If 'question', return a dictionary of format: {
            qkey1: [qkey1_option1, qkey1_option2, ...],
        }
        """
        assert node_type in self.node_types, (
            f"Node type {node_type} not in {self.node_types}."
        )
        indexer: StringIndexer = self.node_labels[node_type]
        labels: List[str] = list(indexer._index.keys())
        if node_type == 'question':
            # special handing of hierarchy: question -> q-o pair
            labels_dict: Dict[str, List[str]] = {}
            for labels in labels:
                qkey = labels.split('_option_')[0].strip()
                labels_dict.setdefault(qkey, []).append(labels)
            return labels_dict
        return labels
    
    def get_edges(self,
                  edge_type: Tuple[str, str, str]
    ) -> List[Tuple[str, str, float]]:
        return list(self.edge_labels[edge_type]._rindex.values())
    
    def get_nodes(self,
                  node_type: str
    ) -> List[str]:
        return list(self.node_labels[node_type]._index.keys())
    
    def negative_edge_sampler(self,
                              source_node_type: str = 'indiv',
                              target_node_type: str = 'question',
    ) -> List[Tuple[str, str]]:
        """
        Custom negative edge sampler is required due to the setup.
        Only if indiv_1 is linked to question_1_option_1,
        (indiv_1, question_1_option_2) is a valid negative edge.
        """
        target_edge_type = None
        for edge_type in self.edge_labels:
            if edge_type[0] == source_node_type \
                and edge_type[2] == target_node_type:
                target_edge_type = edge_type
                break
        assert target_edge_type is not None, (
            f"Nonexisting edge: {source_node_type} -> {target_node_type}."
        )
                
        negative_edges: List[Tuple[str, str]] = []
        source_labels = self.get_labels(source_node_type)
        target_labels = self.get_labels(target_node_type)
        existing_edges = self.edge_labels[target_edge_type]._index.keys()
        for (sl, tl) in existing_edges:
            tl_base = tl.split('_option_')[0].strip()
            tl_items = target_labels[tl_base]
            negative_edges.extend([
                (sl, tl_item)
                for tl_item in tl_items if tl_item != tl
            ])
        for sampled_e in negative_edges:
            if sampled_e in existing_edges:
                raise ValueError(
                    "Should-be negative edge found in existing: "
                    f"{sampled_e}. Data coherence broken."
                )
        return negative_edges
    
    def edge_remover(self,
                     edge_type: Tuple[str, str, str],
                     target_edges: Optional[set[Tuple[str, str]]] = None,
                     fraction: Optional[float] = None,
                     seed: Optional[int] = 42,
                     reverse: bool = False) -> List[Tuple[str, str]]:
        """
        Semantics:
        Remove edges of type edge_type that are either:
            - in target_edges
            - by a randomly sampled fraction of edges
            - note: if fraction is provided, target_edges is ignored
        If reverse is True, remove edges that are NOT in target_edges.
        Returns the removed edges.
        """
        # to prevent incoherent edge removal, constrain graph to unidir.
        self._check_unidirectional()
        existing_map = self.edge_labels[edge_type]._index
        all_pairs: set[Tuple[str, str]] = set(existing_map.keys())
        if target_edges is not None:
            removal_candidates: set = target_edges
        elif fraction is not None:
            assert 0 <= fraction <= 1
            k = int(len(all_pairs) * fraction)
            random.seed(seed)
            removal_candidates = set(random.sample(list(all_pairs), k))
        else:
            raise ValueError("Insufficient arguments to remove edges.")
        removal_set = (
            all_pairs - removal_candidates if reverse
            else removal_candidates & all_pairs
        )
        new_edges: List[Tuple[str, str, float]] = []
        for (src, tgt), (_, weight) in existing_map.items():
            if (src, tgt) not in removal_set:
                new_edges.append((src, tgt, weight))
        self.edge_labels[edge_type] = PairIndexer(new_edges)
        return list(removal_set)
    
    def node_remover(self,
                     node_type: str,
                     target_nodes: Optional[set[str]] = None,
                     fraction: Optional[float] = None,
                     seed: Optional[int] = 42,
                     reverse: bool = False) -> Tuple[
                        List[str],
                        Dict[Tuple[str, str, str], List[Tuple[str, str]]]
                     ]:
        """
        Semantics:
        Remove nodes of type node_type that are either:
            - in target_nodes
            - by a randomly sampled fraction of nodes
            - note: if fraction is provided, target_nodes is ignored
        If edge_also is True, also remove edges that contain these nodes.
        If reverse is True, remove nodes that are NOT in target_nodes.
        """
        # to prevent incoherent node removal, constrain graph to unidir.
        self._check_unidirectional()
        assert node_type in self.node_types, (
            f"Node type {node_type} not in {self.node_types}."
        )
        # safeguard to remove all nodes from same question
        existing_map = self.node_labels[node_type]._index
        all_nodes: set[str] = set(
            qo.split('_option_')[0].strip()
            for qo in existing_map.keys()
        )
        if fraction is not None:
            if fraction == 0.0:
                return [], {}
            assert 0 < fraction <= 1
            k = int(len(all_nodes) * fraction)
            random.seed(seed)
            target_nodes = set(random.sample(list(all_nodes), k))
        if target_nodes is None:
            raise ValueError("Insufficient arguments to remove nodes.")
        target_nodes = [
            node_label.split('_option_')[0].strip()
            for node_label in target_nodes
        ]
        indexer: StringIndexer = self.node_labels[node_type]
        all_nodes = set(indexer._index.keys())
        target_nodes = set([
            node for node in all_nodes
            if any(node.startswith(tn) for tn in target_nodes)
        ])

        # remove nodes from node_labels
        keep_nodes = all_nodes - target_nodes
        if reverse:
            keep_nodes, target_nodes = target_nodes, keep_nodes
        self.node_labels[node_type] = StringIndexer(list(keep_nodes))
        
        # remove edges connected to remove nodes
        target_edges: Dict[
            Tuple[str, str, str], List[Tuple[str, str]]
        ] = {}
        for edge_type in self.edge_labels.keys():
            if edge_type[0] == node_type or edge_type[2] == node_type:
                target_edges[edge_type] = []
                keep_edges = []
                for (from_node, to_node), (_, weight) in \
                    self.edge_labels[edge_type]._index.items():
                    if (from_node not in target_nodes and
                        to_node not in target_nodes):
                        keep_edges.append((from_node, to_node, weight))
                    else:
                        target_edges[edge_type].append(
                            (from_node, to_node)
                        )
                self.edge_labels[edge_type] = PairIndexer(keep_edges)
        return list(target_nodes), target_edges
    
    def graphify(self,
                 bidirectional: bool = True) -> CustomHeteroData:
        
        data = CustomHeteroData()
        for node_type in self.node_types:
            data[node_type].x = None
            # input embedding is determined later
            # can be one-hot encoding, nn.Embedding(one-hot),
            # LLM embedding, trainable LLM embedding, joint training...
        
        for edge_type, edge_indexer in self.edge_labels.items():
            from_type, link, to_type = edge_type
            edges: List[Tuple[int, int ,float]] = []
            for (from_n, to_n), (_, w) in edge_indexer._index.items():
                from_idx = self.node_labels[from_type].get_index(from_n)
                to_idx = self.node_labels[to_type].get_index(to_n)
                edges.append((from_idx, to_idx, w))
            data[edge_type].edge_index = torch.stack([
                torch.tensor(
                    [from_idx for from_idx, _, _ in edges],
                    dtype=torch.long),
                torch.tensor(
                    [to_idx for _, to_idx, _ in edges],
                    dtype=torch.long)
            ])
            data[edge_type].edge_weight = torch.tensor(
                [w for _, _, w in edges], dtype=torch.float32
            )
            if bidirectional:
                data[(to_type, link, from_type)].edge_index = torch.stack([
                    torch.tensor(
                        [to_idx for _, to_idx, _ in edges],
                        dtype=torch.long),
                    torch.tensor(
                        [from_idx for from_idx, _, _ in edges],
                        dtype=torch.long)
                ])
                data[(to_type, link, from_type)].edge_weight = torch.tensor(
                    [w for _, _, w in edges],
                    dtype=torch.float32
                )
        
        data._run_sanity_check()
        return data
    
    def _check_unidirectional(self):
        for edge_type in self.edge_labels:
            from_type, link, to_type = edge_type
            if link == 'self':
                continue
            reverse_edge_type = (to_type, link, from_type)
            assert reverse_edge_type not in self.edge_labels, (
                f"Reverse edge {reverse_edge_type} exists. Not unidirectional."
            )
    
    def __repr__(self):
        repr_str = "GraphInfo("
        repr_str += f"node_types={self.node_types})\n"
        for node_type in self.node_types:
            repr_str += (
                f"  {node_type}: "
                + f"{len(self.get_nodes(node_type))} nodes\n"
            )
        repr_str += f"edge_types={list(self.edge_labels.keys())}\n"
        for edge_type in self.edge_labels.keys():
            repr_str += (
                f"  {edge_type}: "
                + f"{len(self.get_edges(edge_type))} edges\n"
            )
        return repr_str


def build_labels_and_embeddings(
    graphinfo: GraphInfo,
    label_to_x: Dict[str, Dict],
    **kwargs,
) -> Tuple[Dict[str, List[str]], Dict[str, torch.Tensor]]:
    """
    For a given GraphInfo, return (labels, embeddings) dictionaries
    aligned to the graph's node ordering (critical when we prune nodes).
    Return example:
        labels['indiv'][0] = '1001970068'
        embeddings['indiv'][0] = label_to_x['indiv']['1001970068']
    """
    node_types = graphinfo.node_types
    _node_types = set(label_to_x.keys())
    assert node_types == _node_types, (
        f"Node type mismatch: {node_types} vs {_node_types}."
    )
    labels: Dict[str, List[str]] = {
        name: [
            graphinfo.node_labels[name].get_string(i)
            for i in range(len(graphinfo.node_labels[name]))
        ] for name in node_types
    }
    embeddings: Dict[str, torch.Tensor] = {}
    for name in node_types:
        if labels[name] is None:
            embeddings[name] = torch.empty((0,))
        else:
            tensors = [torch.tensor(label_to_x[name][label])
                       for label in labels[name]]
            if tensors[0].dim() == 0:
                tensors = [tensor.unsqueeze(0) for tensor in tensors]
            max_len = max(tensor.shape[0] for tensor in tensors)
            min_len = min(tensor.shape[0] for tensor in tensors)
            if max_len != min_len:
                pad_id = kwargs['tokenizer'].pad_token_id
                tensors = [
                    torch.cat([
                        torch.full((max_len - tensor.shape[0],), pad_id),
                        tensor
                    ]) for tensor in tensors]
            embeddings[name] = torch.stack(tensors)
    assert all(v.dim() == 2 for v in embeddings.values())
    return labels, embeddings


def graphinfo_from_masks(
    base: GraphInfo,
    keep_node_labels: Optional[Dict[str, set[str]]] = None,
    keep_edge_mask: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
) -> GraphInfo:
    """
    Creates a new GraphInfo object by selectively keeping nodes and edges
    based on the provided masks, without modifying the original base GraphInfo.
    The resulting graph maintains structural consistency
    by automatically removing edges that reference any removed nodes.
    Args:
        base: The source GraphInfo instance to filter from.
        keep_node_labels: Optional dictionary mapping node types to sets of node labels
            to retain. If provided, only nodes with labels in these sets are kept.
            Structure: {node_type: {label1, label2, ...}}
            Example: {'indiv': {'1001970068', '1002345678'}, 'question': {'q1_option_1'}}
        keep_edge_mask: Optional dictionary mapping edge types to boolean tensors
            indicating which edges to retain. The tensor indices correspond to the
            edge indexer's _rindex. Structure: {(src_type, rel, tgt_type): torch.Tensor}
            Example: {('indiv', 'responds', 'question'): torch.tensor([True, False, True])}
    Returns:
        A new GraphInfo instance containing only the filtered nodes and edges.
        The new instance has:
        - All node types from the base (even if empty after filtering)
        - Node labels rebuilt with only kept nodes
        - Edge labels rebuilt with only kept edges
        - Edges automatically pruned if they reference removed nodes
    Notes:
        - If keep_node_labels is None for a node type, all nodes of that type are kept
        - If keep_edge_mask is None for an edge type, all edges of that type are kept
        - Edge filtering is applied in two stages:
            1. Apply keep_edge_mask if provided
            2. Remove edges referencing any filtered-out nodes (automatic cleanup)
        - The base GraphInfo remains unchanged (non-destructive operation)
    """
    new_info = object.__new__(GraphInfo)
    new_info.node_types = set(base.node_types)

    new_info.node_labels = {}
    for nt, idxr in base.node_labels.items():
        if keep_node_labels and nt in keep_node_labels:
            labels = [
                s for s in idxr.get_strings()
                if s in keep_node_labels[nt]
            ]
        else:
            labels = idxr.get_strings()
        new_info.node_labels[nt] = StringIndexer(labels)

    new_info.edge_labels = {}
    for et, pi in base.edge_labels.items():
        src, _, tgt = et
        if keep_edge_mask and et in keep_edge_mask:
            mask = keep_edge_mask[et] # mask is [E] bool tensor over pi._rindex
            kept = [
                pi._rindex[i]
                for i in mask.nonzero(as_tuple=False).view(-1).tolist()
            ]
        else:
            kept = list(pi._rindex.values())
        # drop edges that reference removed nodes
        kept_nodes_src = set(new_info.node_labels[src].get_strings())
        kept_nodes_dst = set(new_info.node_labels[tgt].get_strings())
        kept = [
            (s, t, w) for (s, t, w) in kept
            if s in kept_nodes_src and t in kept_nodes_dst
        ]
        new_info.edge_labels[et] = PairIndexer(kept)
    return new_info
