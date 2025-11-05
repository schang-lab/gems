from itertools import chain
from typing import Dict, List, Tuple, Any, Optional, Literal

import torch
from torch import nn
from torch.nn import functional as F

from sibyl.graph.graph import GraphInfo, graphinfo_from_masks
from sibyl.graph.module import CustomHeteroData
from sibyl.utils.misc_utils import rand_indices

ACC_INDIV_CHUNK = 262_144


def profile_memory_heterodata(data: CustomHeteroData) -> None:
    print("=== Node features ===")
    for ntype, x in data.x_dict.items():
        mb = x.element_size() * x.numel() / 1024**2
        print(f"{ntype} | {tuple(x.shape)} | {mb:.3f} MB")
    print("\n=== Edge indices ===")
    for rel, ei in data.edge_index_dict.items():
        mb = ei.element_size() * ei.numel() / 1024**2
        print(f"{str(rel)} | {tuple(ei.shape)} | {mb:.3f} MB")
    if hasattr(data, 'edge_weight_dict'):
        print("\n=== Edge weights ===")
        for rel, ea in data.edge_weight_dict.items():
            mb = ea.element_size() * ea.numel() / 1024**2
            print(f"{str(rel)} | {tuple(ea.shape)} | {mb:.3f} MB")


def prebuild_for_accuracy_logging(
    g_info: GraphInfo,
    edges_to_convert: List[Tuple[str, str]],
):
    """
    Get the question -> question-option indices mapping,
    and source and destination node indices for each question.
    Move to the device of graphinfo.
    Returns:
        qo_idx: Dict[str, torch.Tensor]
            Maps question key to a tensor of indices of options.
        src_by_q: Dict[str, torch.Tensor]
            for each question, stores the list of source node indices
        dst_by_q: Dict[str, torch.Tensor]
            for each question, stores the list of destination node indices
            Note: the unique destination nodes likely equal # options.
    """
    qo_idx: Dict[str, torch.Tensor] = {}
    for qkey, options in g_info.get_labels('question').items():
        idx = [
            g_info.node_labels['question'].get_index(lbl)
            for lbl in options
        ]
        qo_idx[qkey] = torch.tensor(idx, dtype=torch.long,)
    src_by_q: Dict[str, torch.Tensor] = {}
    dst_by_q: Dict[str, torch.Tensor] = {}
    # get source (indiv) and true (question-option) indicies
    for src_lbl, dst_lbl in edges_to_convert:
        qkey = dst_lbl.split('_option_')[0].strip()
        src_by_q.setdefault(qkey, []).append(
            g_info.node_labels['indiv'].get_index(src_lbl)
        )
        dst_by_q.setdefault(qkey, []).append(
            g_info.node_labels['question'].get_index(dst_lbl)
        )
    # make into torch tensor
    for qkey in src_by_q:
        src_by_q[qkey] = torch.tensor(src_by_q[qkey], dtype=torch.long)
        dst_by_q[qkey] = torch.tensor(dst_by_q[qkey], dtype=torch.long)
    return qo_idx, src_by_q, dst_by_q


def prebuild_for_packed_loss(
    n_qo_nodes: int,
    qo_idx: Dict[str, torch.Tensor],
    src_by_q: Dict[str, torch.Tensor],
    dst_by_q: Dict[str, torch.Tensor],
):
    """
    Returns a dict with:
      - src_idx:    [E] individual indices per positive edge
      - tgt_opt:    [E] question-option indices (dst)
      - q_of_edge:  [E] question row id for each edge
      - opt_mat:    [Q, Kmax] right-padded matrix of option indices per question
      - opt_mask:   [Q, Kmax] bool mask for valid columns
      - target_col: [E] column idx of the true option within its question row
    Where:
      Q = number of questions that appear in this split
      Kmax = max #options among those questions
      E = total (indiv, option_true) pairs used for loss calc. in this split
    """

    qkey_list = list(qo_idx.keys())
    Q = len(qkey_list)
    assert Q > 0, "--> prebuild_for_packed_loss: no questions found."

    # Build padded option table [Q, Kmax] with max number of options Kmax
    Kmax = max(int(qo_idx[q].numel()) for q in qkey_list)
    opt_mat  = torch.full((Q, Kmax), -1, dtype=torch.long)
    opt_mask = torch.zeros((Q, Kmax), dtype=torch.bool)

    # Map from question-option node index -> local column within its question
    pos_map = torch.full((n_qo_nodes,), -1, dtype=torch.long)
    # E = total number of indiv - question edges in the current split
    E = sum(int(src_by_q.get(q, torch.empty(0)).numel()) for q in qkey_list)
    src_idx = torch.empty(E, dtype=torch.long)
    tgt_opt = torch.empty(E, dtype=torch.long)
    q_of_edge = torch.empty(E, dtype=torch.long)

    # Fill
    ptr = 0
    for qi, qkey in enumerate(qkey_list):
        options_idx = qo_idx[qkey]
        n_options = int(options_idx.numel())
        if n_options == 0:
            continue
        opt_mat[qi, :n_options] = options_idx
        opt_mask[qi, :n_options] = True
        pos_map[options_idx] = torch.arange(n_options, dtype=torch.long)

        # Edges that answered this question
        s = src_by_q.get(qkey)
        t = dst_by_q.get(qkey)
        if s is None or t is None or s.numel() == 0:
            continue
        n_edges = int(s.numel())
        src_idx[ptr:ptr+n_edges] = s # list of individual indices
        tgt_opt[ptr:ptr+n_edges] = t # list of question-option indices
        q_of_edge[ptr:ptr+n_edges] = qi # question row id
        ptr += n_edges
    assert ptr == E, "--> prebuild_for_packed_loss: number of edges mismatch."

    # Compute target column for each edge in one shot
    target_col = pos_map[tgt_opt]

    return {"src_idx": src_idx,
            "tgt_opt": tgt_opt,
            "q_of_edge": q_of_edge,
            "opt_mat": opt_mat,
            "opt_mask": opt_mask,
            "target_col": target_col,}


def loss_and_acc_over_options_packed(
    h_indiv: torch.Tensor,    # shape = [N_indiv, D]
    h_question: torch.Tensor, # shape = [N_qopt, D]
    decoder: nn.Module,
    pack: Dict[str, torch.Tensor],
    reduction: Literal['unweighted', 'weighted'],
    label_smoothing_alpha: float = 1.0,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Returns:
        loss: scalar tensor (mean CE over all edges)
        correct: int, total number of top-1 correct predictions
        total: int (number of edges)
    """
    device = h_indiv.device
    src_idx    = pack["src_idx"]    # [E]
    q_of_edge  = pack["q_of_edge"]  # [E]
    target_col = pack["target_col"] # [E]
    opt_mat    = pack["opt_mat"]    # [Q, Kmax]
    opt_mask   = pack["opt_mask"]   # [Q, Kmax]

    if src_idx.numel() == 0:
        return torch.tensor(float("nan"), device=device), 0, 0, 0
    assert 0 < label_smoothing_alpha <= 1.0

    E : int = int(src_idx.numel())
    V_by_q : torch.Tensor = h_question[opt_mat.clamp_min(0)] # [Q, Kmax, D]
    # V_by_q represents the question-option embeddings for each question
    is_bilinear : bool = (
        hasattr(decoder, "bilinear")
        and isinstance(decoder.bilinear, nn.Bilinear)
    )
    loss_sum = h_indiv.new_tensor(0.0)
    count = 0
    correct = 0

    for s in range(0, E, ACC_INDIV_CHUNK):
        e = min(s + ACC_INDIV_CHUNK, E)
        U = h_indiv[src_idx[s:e]] # [B, D], individual output reprs
        Vq = V_by_q[q_of_edge[s:e]] # [B, Kmax, D], question-option reprs
        mask = opt_mask[q_of_edge[s:e]].to(device) # [B, Kmax] bool mask
        B, Kmax = Vq.shape[:2]

        if is_bilinear:
            W = decoder.bilinear.weight.squeeze(0) # [D1, D2]
            UiW = U @ W # [B, D2]
            scores = (UiW.unsqueeze(1) * Vq).sum(-1) # [B, Kmax]
            if decoder.bilinear.bias is not None:
                scores = scores + decoder.bilinear.bias
            scores = scores / decoder.tau
        else:
            # Generic fallback (e.g., MLPDecoder) with K-chunking
            D1 = U.size(1)
            kc = 512 if Kmax > 512 else Kmax
            chunks = []
            for ks in range(0, Kmax, kc):
                ke = min(ks + kc, Kmax)
                U_rep = U.unsqueeze(1).expand(-1, ke - ks, -1).contiguous() # [B, kc, D1]
                V_rep = Vq[:, ks:ke, :] # [B, kc, D2]
                sc = decoder.forward(
                    U_rep.reshape(-1, D1), V_rep.reshape(-1, V_rep.size(-1))
                ).reshape(B, -1) # [B, kc]
                chunks.append(sc)
            scores = torch.cat(chunks, dim=1) # [B, Kmax]

        logits = scores.masked_fill(~mask, -1e9)
        targets = target_col[s:e].to(scores.device)

        if abs(label_smoothing_alpha - 1.0) <= 1e-5: # no smoothing
            if reduction == 'unweighted':
                loss_sum = loss_sum + F.cross_entropy(
                    logits, targets, reduction="sum",
                )
            elif reduction == "weighted":
                _loss = F.cross_entropy(
                    logits, targets, reduction="none",
                )
                _weight = 1 / torch.log(mask.sum(dim=1))
                loss_sum = loss_sum + (_weight * _loss).sum() / _weight.mean()
        else:
            logprobs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            logp_target = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
            k_valid = mask.sum(dim=1)
            logprobs_valid = torch.where(
                mask, logprobs, torch.zeros_like(logprobs)
            )
            sum_logp_valid = logprobs_valid.sum(dim=1)
            denom = (k_valid - 1).clamp_min(1).to(logprobs.dtype)
            mean_logp_others = (sum_logp_valid - logp_target) / denom
            _loss = -(
                label_smoothing_alpha * logp_target
                + (1-label_smoothing_alpha) * mean_logp_others
            )
            if reduction == 'unweighted':
                loss_sum = loss_sum + _loss.sum()
            elif reduction == "weighted":
                _weight = 1 / torch.log(k_valid)
                loss_sum = loss_sum + (_weight * _loss).sum() / _weight.mean()

        count += (e - s)
        scores_for_acc = logits.detach()
        pred = scores_for_acc.argmax(dim=1) # [B]
        correct += int((pred == targets).sum().item())

    return loss_sum, count, correct, E


def transductive_split_fast(
    init_graphinfo: GraphInfo,
    edge_types: List[Tuple[str, str, str]],
    fractions: List[float],
    intact_fractions: Dict[str, float],
    seed: int = 42
) -> Tuple[GraphInfo, GraphInfo, Dict[Tuple, List[Tuple]]]:
    """
    Faster version of transductive_split not using deepcopy and negative edge sampler.
    Semantics: remove edges according to fractions.
               Return two graphinfos with removed E, and dictionary of removed E.
               Do not remove edges touching unremovable nodes.
               Unremovable nodes are sampled according to intact_fractions.
    """
    assert len(edge_types) == len(fractions)
    removed_edges: Dict[Tuple, List[Tuple]] = {}
    keep_edge_mask: Dict[Tuple, torch.Tensor] = {}

    unremovable_nodes: Dict[str, set[str]] = {}
    for nt, frac in intact_fractions.items():
        if frac <= 0.0:
            continue
        unremovable_nodes[nt] = _sample_node_labels_grouped(
            ginfo=init_graphinfo,
            node_type=nt,
            frac=frac,
            seed=seed
        )
    for et, frac in zip(edge_types, fractions):
        pi = init_graphinfo.edge_labels[et] # pairindexer
        E = len(pi._rindex)
        keep_mask = torch.ones(E, dtype=torch.bool)
        if E == 0 or frac <= 0.0:
            keep_edge_mask[et] = torch.ones(E, dtype=torch.bool)
            removed_edges[et] = []
            continue

        src_nt, _, dst_nt = et
        unrem_src = unremovable_nodes.get(src_nt, set())
        unrem_dst = unremovable_nodes.get(dst_nt, set())
        eligible_idx = torch.tensor([
            i for i, (s, t, w) in pi._rindex.items()
            if s not in unrem_src and t not in unrem_dst
        ], dtype=torch.long)
        n_eligible = eligible_idx.numel()
        if int(n_eligible * frac) == 0:
            keep_edge_mask[et] = torch.ones(E, dtype=torch.bool)
            removed_edges[et] = []
            continue

        removed_idx = eligible_idx[rand_indices(n_eligible, frac, seed=seed)]
        keep_mask[removed_idx] = False
        keep_edge_mask[et] = keep_mask
        removed_edges[et] = [tuple(pi._rindex[i][:2]) for i in removed_idx.tolist()]
        # _rindex is a dict of {idx: (src, dst, weight)}

    final_graphinfo = graphinfo_from_masks(
        base=init_graphinfo,
        keep_node_labels=None,
        keep_edge_mask=keep_edge_mask
    )
    init_graphinfo = graphinfo_from_masks(
        base=init_graphinfo,
        keep_node_labels=None,
        keep_edge_mask=keep_edge_mask
    )
    return init_graphinfo, final_graphinfo, removed_edges


def inductive_split_fast(
    init_graphinfo: GraphInfo,
    node_types: List[str],
    fractions: List[float],
    preselected_nodes: Dict[str, set[str]] = {},
    seed: int = 42
) -> Tuple[GraphInfo, GraphInfo, Dict[str, set[str]], Dict[Tuple, List[Tuple]]]:
    """
    Faster version of inductive_split.
    Semantics: remove nodes according to fractions and touched edges.
               preselected_nodes are removed instead of sampled.
    Returns:
        init_graphinfo: GraphInfo with removed nodes.
        final_graphinfo: GraphInfo with removed indiv-question edges
    """
    assert len(node_types) == len(fractions)
    removed_nodes: Dict[str, set[str]] = {}
    for nt, frac in zip(node_types, fractions):
        if preselected_nodes.get(nt) is None:
            removed_nodes[nt] = _sample_node_labels_grouped(
                ginfo=init_graphinfo,
                node_type=nt,
                frac=frac,
                seed=seed
            )
        else:
            removed_nodes[nt] = preselected_nodes[nt]
    removed_edges: Dict[Tuple, List[Tuple[str, str]]] = {}
    for et, pi in init_graphinfo.edge_labels.items():
        src_nt, _, dst_nt = et
        touched = []
        for i, (s, t, w) in pi._rindex.items():
            if (src_nt in removed_nodes and s in removed_nodes[src_nt]) or \
               (dst_nt in removed_nodes and t in removed_nodes[dst_nt]):
                touched.append((s, t))
        removed_edges[et] = list(set(touched))
    keep_node_labels = {
        nt: set(init_graphinfo.node_labels[nt].get_strings()) - removed_nodes[nt]
        for nt in removed_nodes
    }
    # keep_edge_mask: remove only indiv - question edges for final_graphinfo
    # but keep subgroup - indiv edges.
    keep_edge_mask: Dict[Tuple, torch.Tensor] = {}
    for et, pi in init_graphinfo.edge_labels.items():
        if et == ('indiv', 'responds', 'question'):
            removed_idx = [pi.get_index(e)[0] for e in removed_edges[et]]
            keep_mask = torch.ones(len(pi._rindex), dtype=torch.bool)
            keep_mask[removed_idx] = False
            keep_edge_mask[et] = keep_mask
    # 1) build final_graphinfo, which has removed indiv - question edges
    # and but keeps all nodes. Nodes will be removed at init_graphinfo.
    final_graphinfo = graphinfo_from_masks(
        base=init_graphinfo,
        keep_node_labels=None,
        keep_edge_mask=keep_edge_mask,
    )
    # 2) from init_graphinfo, remove nodes in removed_nodes
    init_graphinfo = graphinfo_from_masks(
        base=init_graphinfo,
        keep_node_labels=keep_node_labels,
        keep_edge_mask=None
    )
    return init_graphinfo, final_graphinfo, removed_nodes, removed_edges


def split_based_on_predefined(
    init_graphinfo: GraphInfo,
    predefined_split_info: Dict[str, Dict[str, Dict[str, float]]],
) -> Tuple[GraphInfo, GraphInfo, Dict[str, set[str]], Dict[Tuple, List[Tuple]]]:
    """
    Train / val / test split based on predefined split info.
    """
    assert set(predefined_split_info.keys()) == {'keep', 'remove'}, (
        "predefined_split_info must have exactly two keys: 'keep' and 'remove'"
    )

    # search over 'keep' to get individuals and questions to keep
    kept_nodes: Dict[str, set[str]] = {'indiv': set(), 'question': set()}
    for indiv, q in predefined_split_info['keep'].items():
        qkeys_set = set(q.keys())
        if qkeys_set != set():
            kept_nodes['indiv'].add(indiv)
            kept_nodes['question'].update(qkeys_set)

    # search over 'remove'. if indiv or question not in kept_nodes, add to removed_nodes
    # also collect removed edges
    removed_nodes: Dict[str, set[str]] = {'indiv': set(), 'question': set()}
    removed_edges: Dict[Tuple, List[Tuple[str, str]]] = {}
    for indiv, q in predefined_split_info['remove'].items():
        removed_nodes['indiv'].add(indiv)
        removed_nodes['question'].update(set(q.keys()))
        for qkey, choice in q.items():
            removed_edges.setdefault(
                ('indiv', 'responds', 'question'), []
            ).append((indiv, qkey + f"_option_{int(choice)}"))
    removed_nodes['indiv'] = removed_nodes['indiv'] - kept_nodes['indiv']
    removed_nodes['question'] = removed_nodes['question'] - kept_nodes['question']

    # define keep_node_labels and keep_edge_mask, as in inductive/transductive_split_fast
    keep_node_labels: Dict[str, set[str]] = {}
    for nt in removed_nodes:
        original_set = set(init_graphinfo.node_labels[nt].get_strings())
        if nt == 'indiv':
            assert removed_nodes[nt].issubset(original_set)
            keep_node_labels[nt] = original_set - removed_nodes[nt]
        elif nt == 'question':
            for qkey in removed_nodes[nt]:
                assert any(original.startswith(qkey) for original in original_set)
            keep_node_labels[nt] = set(
                original for original in original_set
                if not any(original.startswith(qkey) for qkey in removed_nodes[nt])
            )

    keep_edge_mask: Dict[Tuple, torch.Tensor] = {}
    et = ('indiv', 'responds', 'question')
    if et in init_graphinfo.edge_labels:
        pi = init_graphinfo.edge_labels[et]
        removed_idx = [pi.get_index(e)[0] for e in removed_edges[et]]
        keep_mask = torch.ones(len(pi._rindex), dtype=torch.bool)
        keep_mask[removed_idx] = False
        keep_edge_mask[et] = keep_mask

    final_graphinfo = graphinfo_from_masks(
        base=init_graphinfo,
        keep_node_labels=None,
        keep_edge_mask=keep_edge_mask,
    )
    init_graphinfo = graphinfo_from_masks(
        base=init_graphinfo,
        keep_node_labels=keep_node_labels,
        keep_edge_mask=keep_edge_mask,
    )
    return init_graphinfo, final_graphinfo, removed_nodes, removed_edges


def _sample_node_labels_grouped(
    ginfo: GraphInfo, node_type: str, frac: float, seed: int = 42
) -> set[str]:
    """
    For 'indiv': sample labels directly.
    For 'question': sample qkeys and include ALL options for each sampled qkey.
    Returns a set of FULL labels to remove (e.g., 'TITLE9_W103_option_2', ...).
    """
    if frac <= 0.0:
        return set()
    g = torch.Generator().manual_seed(seed)
    if node_type == 'question': # special handling
        qdict = ginfo.get_labels('question')
        qkeys = list(qdict.keys())
        k = int(len(qkeys) * frac)
        if k <= 0:
            return set()
        idx = torch.randperm(len(qkeys), generator=g)[:k].tolist()
        return set(chain.from_iterable(qdict[qkeys[i]] for i in idx))
    else:
        labels = ginfo.get_nodes(node_type)
        n = len(labels)
        k = int(n * frac)
        if k <= 0:
            return set()
        idx = torch.randperm(n, generator=g)[:k].tolist()
        return set(labels[i] for i in idx)


def run_forward_pass(
    embeddings: Dict[str, torch.Tensor],
    encoders: Dict[Tuple[str, str], nn.Module],
    graph,
    graphnn,
    link_pred_module: Dict[str, nn.Module],
    epoch: int,
    split_name: str,
    ce_pack: Dict[str, Any],
    device: torch.device,
    loss_weighting: Dict[str, float],
    reduction: Literal['unweighted', 'weighted'],
    label_smoothing_alpha: float,
):
    """
    Args:
        reduction:
            'unweighted': CE loss is averaged equally
            'weighted': CE loss is weighted by log(#options)
        label_smoothing_alpha:
            Exclusive smoothing alpha for label smoothing in CE loss
            Correct label assigned alpha, other labels assigned (1-alpha)/(n-1)
        loss_weighting: Dict[str, float]
            Especially during training, weight different loss components
            (e.g. loss from inductive learning vs. transductive learning) differently.
            loss_weighting normalized to sum to 1.0
        dist_weighting: float
            Two loss components (indiv-question, subgroup-question) are combined
    """
    assert reduction in ['unweighted', 'weighted']
    assert set(ce_pack.keys()) == set(loss_weighting.keys())
    loss_weighting_sum = sum(loss_weighting.values()) / len(loss_weighting)
    loss_weighting = {k: v / loss_weighting_sum for k, v in loss_weighting.items()}

    # forward pass
    for name, encoder in encoders.items():
        graph[name].x = encoder(embeddings[name].to(device))
    out = graphnn(
        x_dict=graph.x_dict,
        edge_index_dict=graph.edge_index_dict,
    )
    
    # individual - question link prediction by CE
    indiv_loss = torch.tensor(0.0, device=device)
    total_len_indiv = 0
    correct_all, total_all = 0, 0

    for pe_name in ce_pack.keys():
        if ce_pack[pe_name]['src_idx'].size(0) == 0: # no edges to compute loss
            continue
        loss, count, correct, total = loss_and_acc_over_options_packed(
            h_indiv=out['indiv'],
            h_question=out['question'],
            decoder=link_pred_module['indiv_question'],
            pack=ce_pack[pe_name],
            reduction=reduction,
            label_smoothing_alpha=label_smoothing_alpha,
        )
        assert not torch.isnan(loss), "Loss is NaN"
        print(f"Epoch {epoch:03d} | {split_name} {pe_name} kl loss {(loss.item()/count):.4f}")
        print(f"Epoch {epoch:03d} | {split_name} {pe_name} individual acc {(correct/total):.4f}")
        indiv_loss += loss * loss_weighting[pe_name]
        total_len_indiv += count
        correct_all += correct
        total_all += total

    # total loss: weighted sum of individual loss and subgroup loss
    total_loss = indiv_loss / total_len_indiv
    return total_loss, out, correct_all, total_all