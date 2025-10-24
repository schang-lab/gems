import copy
import datetime
import os
import json
import random
import pathlib
from itertools import chain
from typing import List, Dict, Tuple, Union, Any

import hydra
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from sibyl.graph.node import (
    NodeCollection,
    nodify_all,
    map_label_to_x,
)
from sibyl.graph.graph import (
    GraphInfo,
    build_labels_and_embeddings,
)
from sibyl.graph.module import (
    OneHotAttributeEncoder, OneHotEncoder, FixedEmbedding,
    GeneralHeteroGraphConv, GeneralHeteroGAT, GeneralHeteroSAGE, GeneralHeteroRGCNConv,
    BilinearDecoder, MLPDecoder,
)
from sibyl.graph.graph_utils import (
    prebuild_for_accuracy_logging, prebuild_for_packed_loss,
    inductive_split_fast, transductive_split_fast, split_based_on_predefined,
    run_forward_pass,
)
from sibyl.graph.preproc_cache import (
    compute_prep_cache_key,
    try_load_prep_cache,
    save_prep_cache,
    move_prep_to_device,
)
from sibyl.utils.logger import start_capture
from sibyl.config.dataset_map import DATASET_MAP


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _all_zero_grad(optimizers: Dict[str, Any]):
    for _optim in optimizers.values():
        if isinstance(_optim, dict):
            _all_zero_grad(_optim)
        else:
            _optim.zero_grad()
    return

def _optim_all_step(optimizers: Dict[str, Any]):
    for _optim in optimizers.values():
        if isinstance(_optim, dict):
            _optim_all_step(_optim)
        else:
            _optim.step()

def _sched_all_step(schedulers: Dict[str, Any]):
    for _sched in schedulers.values():
        if isinstance(_sched, dict):
            _sched_all_step(_sched)
        else:
            _sched.step()


@hydra.main(
    config_path=os.path.join(REPO_ROOT, "sibyl", "config", "graph"),
    config_name="gems_training_config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    device = torch.device('cuda') if torch.cuda.is_available() else None
    assert device is not None, "CUDA is not available. Abort"

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.experiment.use_logger:
        log_path = os.path.join(
            REPO_ROOT, "outputs", "logs", f"gems_training",
            f"log_{datetime_str}.log"
        )
        print(f"--> Starting logger to {str(log_path)}")
        _ = start_capture(
            debug=True,
            save_path=log_path,
        )
    print(f"--> Running with config: {cfg}")


    ###############################################
    ##### Load and parse hydra configurations #####
    ###############################################
    cfg_pynative = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg_exp = cfg.experiment
    assert cfg_exp.save_criterion in ["acc", "loss"], (
        "Save criterion must be KL-loss or accuracy. "
        "Transductive setting refers to setting 1 (missing responses, i.e. imputation). "
        "Inductive setting refers to setting 2 (new individuals) and 3 (new questions). "
        "By default, we use *_acc (i.e. validation acc) for checkpoint selection."
    )

    cfg_data = cfg_pynative['dataset']
    dataset_name: str = cfg_data['dataset_name']
    dataset_info: Dict[str, Any] = DATASET_MAP[dataset_name]
    exit_undefined_traits: bool = cfg_data['exit_undefined_traits']
    # if exit_undefined_traits = True,
    # individuals with at least on missing trait (e.g. age is missing) are skipped
    node_types: List[str] = cfg_data['node_types']
    # by default, three node types: 'indiv', 'subgroup', 'question' (refer to Fig. 1)
    filtering_max_indiv = cfg_data['filtering']['n_indiv_sample_per_wave']
    # samples at most filtering_max_indiv individuals. useful during dev
    predefined_split: Dict = json.load(
        open(os.path.join(REPO_ROOT, cfg_data['split_filepath']), 'r')
    )
    if cfg.split_info.new_question.is_this:
        # flag of Setting 3 (new questions), which does transductive training during GNN
        # so there predefined_split is not used as it is;
        # instead, a train data of predefined_split is further split into train/val        
        random.seed(cfg.training.seed)
        
        def split_dict(d: Dict, p: float) -> Tuple[Dict, Dict]:
            items = list(d.items())
            random.shuffle(items)
            split_idx = int(len(items) * (1-p))
            return dict(items[:split_idx]), dict(items[split_idx:])

        _tmp_dict = {'train': {}, 'val': {},
                     'test': predefined_split['test']} # test doesn't matter here
        for indiv_id, indiv_response in predefined_split['train'].items():
            train_qs, val_qs = split_dict(indiv_response,
                                          cfg.split_info.new_question.fraction)
            _tmp_dict['train'][indiv_id] = train_qs
            _tmp_dict['val'][indiv_id] = val_qs
        predefined_split = {**_tmp_dict}

    for split_name in predefined_split:
        predefined_split[split_name] = {
            str(int(float(k))): v for k, v in predefined_split[split_name].items()
        }
    # predefined split info generated from run_dataset_split.py
    # don't be confused with hydra config 'split_info' below.
    # hydra 'split_info' determines how to split message passing / supervision edges
    # predefined_split determines which individuals / questions go to train/val/test sets

    cfg_gnn = cfg_pynative['gnn']
    gnn_arch: str = cfg_gnn['gnn_arch']
    assert gnn_arch in ["graphconv", "gat", "sage", "rgcn"], (
        f"Not supported GNN architecture: {gnn_arch}. "
        f"You can implement it in sibyl.graph.module.. We used RGCN, GAT, and GraphSAGE"
    )
    gnn_output_embed_dim: Dict[str, int] = cfg_gnn['gnn_output_embed_dim']
    gnn_num_layers: int = cfg_gnn['gnn_num_layers']
    gnn_hidden_dims: Dict[str, List[int]] = cfg_gnn['gnn_hidden_dims']
    gnn_nonlin_fn: Dict[str, str] = cfg_gnn['nonlin_fn']
    gnn_p_dropout: float = cfg_gnn['gnn_p_dropout']
    aggr_method: str = cfg_gnn['aggr_method']
    hetero_aggr_method: str = cfg_gnn['hetero_aggr_method']
    add_self_loops: bool = cfg_gnn['add_self_loops']
    # we set add_self_loops = True for GAT, and = False for others
    norm_type: str = cfg_gnn['norm_type']
    use_residual: bool = cfg_gnn['residual']
    if gnn_arch == "gat": # GAT-specific parameters
        gnn_n_heads: List[int] = cfg_gnn['gnn_n_heads']
        concat: bool = cfg_gnn['concat']
        attn_dropout: float = cfg_gnn['attn_dropout']
        negative_slope: float = cfg_gnn['negative_slope']

    cfg_train = cfg.training
    split_info = cfg.split_info
    cfg_encoder = cfg_pynative['encoder']
    # in cfg_encoder, define the embedding dim. and which encoder to use for each node type
    gnn_input_embed_dim: Dict[str, int] = cfg_encoder['gnn_input_embed_dim']
    embedding_scheme: Dict[str, str] = cfg_encoder['encoding']

    cfg_decoder = cfg_pynative['decoder']
    decoder_types = cfg_decoder.keys()
    for _type in decoder_types:
        assert cfg_decoder[_type]['decoder_arch'] in ["bilinear", "mlp", "dot_product"], (
            f"Decoder architecture {_type} not supported. "
            f"Please choose from: bilinear, mlp, dot_product."
        )

    if cfg_exp.save_model:
        save_path_model = os.path.join(
            REPO_ROOT, "outputs", "gems_models", f"gems_training_{gnn_arch}",
            f"best_model_{datetime_str}.pth"
        )
        os.makedirs(os.path.dirname(save_path_model), exist_ok=True)
        print(f"--> Save path for model checkpoint: {save_path_model}")
    if cfg_exp.save_embedding:
        save_path_embed = os.path.join(
            REPO_ROOT, "outputs", "gems_embeddings", f"gems_training_{gnn_arch}",
            f"best_embedding_{datetime_str}.pth"
        )    
        os.makedirs(os.path.dirname(save_path_embed), exist_ok=True)        
        print(f"--> Save path for embedding: {save_path_embed}")

    # compute_prep_cache_key computes a unique key based on the configuration
    cache_key, cache_sig = compute_prep_cache_key(cfg_pynative)
    # try to load from prep cache, unless environ_var GEMS_FORCE_RECACHE is set to "1"
    bundle = (
        None if (os.environ.get("GEMS_FORCE_RECACHE", "0") == "1")
        else try_load_prep_cache(REPO_ROOT, cache_key)
    )
    if bundle is not None:
        print(f"--> Found prep cache (key={cache_key}).")
    else:
        print(f"--> No prep cache found or not using prep cache (key={cache_key}).")


    ##########################################################
    ##### Create train/val/test graph ready for training #####
    ##########################################################
    # either failed to load graph from disk or loading is skipped; need to (re)create graph
    if bundle is None:

        global_container: Dict[str, Any] = {}

        print(f"--> Creating global NodeCollection ...")
        global_nodecollection = NodeCollection(
            source=dataset_info['source'],
            basepath=REPO_ROOT / "data" / dataset_name,
            qkeypath=dataset_info['qkeypath'],
            qstringpath=dataset_info['qstringpath'],
            valid_traits=dataset_info['valid_traits'],
            use_selective_subgroups=dataset_info['use_selective_subgroups'],
            min_questions_per_indiv=dataset_info['min_questions_per_indiv'],
            dataset_name=dataset_name,
        )

        print("--> Nodify all individuals, subgroups, questions ...")
        global_all_nodes = nodify_all(
            collection=global_nodecollection,
            embedding_scheme=embedding_scheme,
            add_self_loops=add_self_loops,
            node_types=node_types,
            exit_undefined=exit_undefined_traits,
        )
        for nt in global_all_nodes.keys():
            print(f"--> [global] {nt} node numbers: {len(global_all_nodes[nt])}")

        print("--> Creating global map from label to x (input node feature)...")
        global_label_to_x: Dict[str, Dict[str, Union[List, int]]] = {
            name: map_label_to_x(global_all_nodes[name])
            for name in global_all_nodes
        }
        # global_label_to_x structure:
        # { node_type: { label_str: input_representation } }
        # e.g., { 'indiv' : { '2019010683680082' : [0, 1, 0, ..., 5] }, ... }
        # e.g., { 'question' : { 'Q1_option_1.0' : 42, ... }, ... }
        
        print("--> Building per-wave train/val/test splits...")
        # some human simulation datasets have multiple waves, like opinionQA
        # we build separate graphs for each wave, and train a single model on multiple graphs
        wave_numbers = [int(qkey.split("_W")[-1]) for qkey in global_nodecollection.qkeys]
        wave_numbers = sorted(list(set(wave_numbers)))
        _idx = 0
        for _wave in wave_numbers:
            wave_prefix = f"wave{_wave}"
            print(f"=============== Building splits for wave {_wave} ===============")

            wave_nodecollection = NodeCollection(
                wave_number=[_wave],
                source=dataset_info['source'],
                basepath=REPO_ROOT / "data" / dataset_name,
                qkeypath=dataset_info['qkeypath'],
                qstringpath=dataset_info['qstringpath'],
                valid_traits=dataset_info['valid_traits'],
                use_selective_subgroups=dataset_info['use_selective_subgroups'],
                min_questions_per_indiv=dataset_info['min_questions_per_indiv'],
                n_indiv_sample_per_wave=filtering_max_indiv,
                make_subgroup_nodes=('subgroup' in node_types),
                dataset_name=dataset_name,
            )
            wave_nodes: Dict[str, List[Dict]] = nodify_all(
                collection=wave_nodecollection,
                embedding_scheme=embedding_scheme,
                add_self_loops=add_self_loops,
                node_types=node_types,
                exit_undefined=exit_undefined_traits,
            )
            for nt in wave_nodes.keys():
                print(f" --> [{wave_prefix}] {nt} node numbers: {len(wave_nodes[nt])}")
            if len(wave_nodes['indiv']) == 0:
                print(f" --> [{wave_prefix}] No individuals in this wave. Skipping.")
                continue

            print(f" --> Building GraphInfo object for {wave_prefix} ...")
            wave_graphinfo = GraphInfo(nodes=list(chain.from_iterable(wave_nodes.values())))

            # note that we use the predefined_split to ensure same data split for LLM / GEMS exps.
            # during initial dev, one can use fractions / seed info to randomly train/val/test split
            # using utils/graph/graph_utils.py inductive_split_fast and transductive_split_fast,
            # but not recommended for final eval when comparing with LLM-based methods.

            if predefined_split != {}:
                # predefined_wave_split contains wave-specific split information
                predefined_wave_split = {
                    split_name: {
                        indiv: {
                            qkey: choice for qkey, choice in q_list.items()
                            if qkey.endswith(f"_W{_wave}")
                        } for indiv, q_list in predefined_split[split_name].items()
                    } for split_name in predefined_split
                }
                for split_name in predefined_wave_split:
                    for indiv in list(predefined_wave_split[split_name].keys()):
                        if predefined_wave_split[split_name][indiv] == {}:
                            predefined_wave_split[split_name].pop(indiv)
                # build test split
                wave_graphinfo, ginfo_test, rnodes_t, redges_t = split_based_on_predefined(
                    init_graphinfo=wave_graphinfo,
                    predefined_split_info={
                    'keep': {
                        k : {
                            **predefined_wave_split['train'][k],
                            **predefined_wave_split['val'].get(k, {})
                        } for k in predefined_wave_split['train']
                    },
                    'remove': predefined_wave_split['test']},
                )
                global_container[f'{wave_prefix}_test'] = {
                    'graphinfo': ginfo_test,
                    'pos_edges': redges_t['indiv', 'responds', 'question'],
                    'removed_nodes': rnodes_t,
                    'removed_edges': redges_t,
                }
                # build val split
                wave_graphinfo, ginfo_val, rnodes_v, redges_v = split_based_on_predefined(
                    init_graphinfo=wave_graphinfo,
                    predefined_split_info={
                    'keep': predefined_wave_split['train'],
                    'remove': predefined_wave_split['val']},
                )
                global_container[f'{wave_prefix}_val'] = {
                    'graphinfo': ginfo_val,
                    'pos_edges': redges_v['indiv', 'responds', 'question'],
                    'removed_nodes': rnodes_v,
                    'removed_edges': redges_v,
                }
            else:
                raise ValueError("Predefined split info is required for reproducible experiments.")
        
            ### ensure same eval setting with LLM-based methods
            # Setting 1 (imputation)
            if "evalpartial_0p00" not in cfg.dataset.split_filepath:
                global_container[f'{wave_prefix}_test']['graphinfo'] = copy.deepcopy(
                    global_container[f'{wave_prefix}_val']['graphinfo']
                ) # use the same graphinfo for val and test
            # Setting 3 (new questions)
            elif "_individual_" not in cfg.dataset.split_filepath:
                _, global_container[f'{wave_prefix}_test']['graphinfo'], _, _ = inductive_split_fast(
                    global_container[f'{wave_prefix}_test']['graphinfo'],
                    node_types=['question'],
                    fractions=[0.0],
                    preselected_nodes=global_container[f'{wave_prefix}_val']['removed_nodes'],
                ) # do not allow validation questions to appear in the test graph
            # Setting 2 (new individuals)
            else:
                _, global_container[f'{wave_prefix}_test']['graphinfo'], _, _ = inductive_split_fast(
                    global_container[f'{wave_prefix}_test']['graphinfo'],
                    node_types=['indiv'],
                    fractions=[0.0],
                    preselected_nodes=global_container[f'{wave_prefix}_val']['removed_nodes'],
                ) # do not allow validation individuals to appear in the test graph

            for _ in range(split_info.graphsaint.n_sample):
                # first, subsample nodes from the train graph.
                # theoretically it can be used when there are too many individual nodes.
                # default to fraction=1.0,
                # and in this case node_sampled identical to wave_graphinfo (train graph)
                node_sampled, _, _, _ = inductive_split_fast(
                    wave_graphinfo, node_types=['indiv'],
                    fractions=[1-split_info.graphsaint.fraction],
                    seed=_idx + cfg_train.seed,
                )
                # second, remove some nodes (and their touching edges)
                # for setting 1,3 : do not remove any nodes from train graph,
                # because it is purely transductive setting
                #  --> split_info.inductive.train.indiv = 0.0
                #  --> split_info.inductive.train.question = 0.0
                # - note: even though setting 3 is new question setting,
                #   during GNN training we aim to learn representations of individuals and train questions
                #   so we don't have to remove any nodes during GNN training.
                # for setting 2, remove some individual nodes
                #  --> split_info.inductive.train.indiv = 0.5 (refer to experiemnt section)
                #  --> split_info.inductive.train.question = 0.0
                _, train_ginfo, rnodes_train_ind, redges_train_ind = inductive_split_fast(
                    node_sampled, node_types=['indiv', 'question'],
                    fractions=[
                        split_info.inductive.train.indiv,
                        split_info.inductive.train.question,
                    ],
                    seed=_idx + cfg_train.seed,
                )
                # third, remove some edges from the remaining train graph
                train_ginfo, _, redges_train_trans = transductive_split_fast(
                    train_ginfo, edge_types=[('indiv', 'responds', 'question')],
                    fractions=[split_info.transductive.train.indiv_question],
                    intact_fractions={
                        'indiv': split_info.transductive.train.intact_indiv,
                    },
                    seed=_idx + cfg_train.seed,
                )
                train_split_name = f'train_{_idx}'
                global_container[train_split_name] = {
                    'graphinfo': train_ginfo,
                    'inductive_pos_edges': redges_train_ind['indiv', 'responds', 'question'],
                    'transductive_pos_edges': redges_train_trans['indiv', 'responds', 'question'],
                    'removed_nodes': rnodes_train_ind,
                    'removed_edges': redges_train_ind,
                }
                _idx += 1
            print(f" --> {wave_prefix} {split_info.graphsaint.n_sample} train graphinfo(s) built.")

        print(f"--> GraphInfo(s) all created.")
        for split in sorted(global_container.keys()):
            print(f"--> {split} edges for performance measurement: ")
            if "train" not in split: # evaluation graphs: 'pos_edges' contain edges to predict
                print(f"{len(global_container[split]['pos_edges'])}")
            else: # train graphs
                # one of the two types of positive edges will be empty, depending on the setting.
                print(f"Inductive setting: {len(global_container[split]['inductive_pos_edges'])}")
                print(f"Transductive setting: {len(global_container[split]['transductive_pos_edges'])}")
            print(f"--> {split} graphinfo: "
                  f"{global_container[split]['graphinfo']}")

        ### the following lines are preprocessing for batched training and fast evaluation.
        print("--> Creating index-to-label, index-to-x mappings...")
        graph_labels: Dict[str, Dict[str, List[str]]] = {}
        graph_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        for split in tqdm(global_container.keys(), desc="index -> label, index -> embedding"):
            graph_labels[split], graph_embeddings[split] = build_labels_and_embeddings(
                graphinfo=global_container[split]['graphinfo'],
                label_to_x=global_label_to_x,
            )
        print("--> Index-to-label, index-to-x (embedding) mappings created.")

        print("\n--> Prebuilding tensors and graphs ...")
        qo_idx:    Dict = {split: {} for split in global_container.keys()}
        src_by_q:  Dict = {split: {} for split in global_container.keys()}
        dst_by_q:  Dict = {split: {} for split in global_container.keys()}
        ce_packs:  Dict = {split: {} for split in global_container.keys()}
        graphs:    Dict = {} # there is only one graph per split
        
        for split in tqdm(global_container.keys(), desc="prebuilding preproc tensors and graphs"):
            # prebuild the edge indicies of positive edges, grouped by qkey
            for positive_edge_name in global_container[split].keys():
                if not positive_edge_name.endswith('pos_edges'):
                    continue
                qo, sbq, dbq = prebuild_for_accuracy_logging(
                    g_info=global_container[split]['graphinfo'],
                    edges_to_convert=global_container[split][positive_edge_name],
                )
                qo_idx[split][positive_edge_name] = qo
                src_by_q[split][positive_edge_name] = sbq
                dst_by_q[split][positive_edge_name] = dbq
                ce_packs[split][positive_edge_name] = prebuild_for_packed_loss(
                    n_qo_nodes=len(global_container[split]['graphinfo'].node_labels['question']),
                    qo_idx=qo,
                    src_by_q=sbq,
                    dst_by_q=dbq,
                )
            graphs[split] = global_container[split]['graphinfo'].graphify()

        print("--> Prebuild complete.")
        bundle = {
            "global_all_nodes": global_all_nodes,
            "global_label_to_x": global_label_to_x,
            "global_container": global_container,
            "graph_labels": graph_labels,
            "graph_embeddings": graph_embeddings,
            "qo_idx": qo_idx,
            "src_by_q": src_by_q,
            "dst_by_q": dst_by_q,
            "ce_packs": ce_packs,
            "graphs": graphs,
        }
        if cfg_exp.save_data:
            print(f"--> Saving prep cache (key={cache_key}) ...")
            save_prep_cache(REPO_ROOT, cache_key, cache_sig, bundle)
            print("--> Prep cache saved.")

    else:
        global_all_nodes = bundle["global_all_nodes"]
        global_label_to_x = bundle["global_label_to_x"]
        global_container = bundle["global_container"]
        graph_labels = bundle["graph_labels"]
        graph_embeddings = bundle["graph_embeddings"]
        qo_idx = bundle["qo_idx"]
        src_by_q = bundle["src_by_q"]
        dst_by_q = bundle["dst_by_q"]
        ce_packs = bundle["ce_packs"]
        graphs = bundle["graphs"]

    # loss weighting.
    # we initially tested with different weights, but eventually settled with 1.0 for all settings.
    loss_w: Dict = {split: {} for split in global_container.keys()}
    for split in global_container.keys():
        for positive_edge_name in global_container[split].keys():
            if not positive_edge_name.endswith('pos_edges'):
                continue
            loss_w[split][positive_edge_name] = (
                cfg.training.inductive_loss_weight if 'inductive' in positive_edge_name
                else cfg.training.transductive_loss_weight if 'transductive' in positive_edge_name
                else 1.0
            )

    #############################################################################################
    ##### Create NN modules - input embedding encoders, graph encoders, decoders (Figure 2) #####
    #############################################################################################
    print("--> Creating encoders ...")
    encoder : Dict[str, nn.Module] = {}

    if 'indiv' in node_types:
        if embedding_scheme['indiv'] == 'trait':
            encoder['indiv'] = OneHotAttributeEncoder(
                embedding_dim=gnn_input_embed_dim['indiv'],
                attr_list=dataset_info['valid_traits'],
                source=dataset_info['source'],
            )
        elif embedding_scheme['indiv'] == 'fixed':
            encoder['indiv'] = FixedEmbedding()
        elif embedding_scheme['indiv'] == 'one_hot':
            encoder['indiv'] = OneHotEncoder(
                input_dim=len(global_all_nodes['indiv']),
                embedding_dim=gnn_input_embed_dim['indiv'],
                set_trainable=True,
            )
        else:
            raise ValueError(f"Unsupported indiv embedding scheme: {embedding_scheme['indiv']}")

    if 'question' in node_types:
        if embedding_scheme['question'] in ['one_hot', 'random']:
            encoder['question'] = OneHotEncoder(
                input_dim=len(global_all_nodes['question']),
                embedding_dim=gnn_input_embed_dim['question'],
                set_trainable=(embedding_scheme['question'] != 'random')
            )
        else:
            raise ValueError(f"Unsupported question embedding scheme: {embedding_scheme['question']}")

    if 'subgroup' in node_types:
        if embedding_scheme['subgroup'] == 'one_hot':
            encoder['subgroup'] = OneHotEncoder(
                input_dim = max([node['x'] for node in global_all_nodes['subgroup']]) + 1,
                embedding_dim=gnn_input_embed_dim['subgroup']
            )
        elif embedding_scheme['subgroup'] == 'fixed':
            encoder['subgroup'] = FixedEmbedding()
        else:
            raise ValueError(f"Unsupported subgroup embedding scheme: {embedding_scheme['subgroup']}")

    print("--> Creating graph encoder and dry run...")
    with torch.no_grad():
        dummy_split = list(global_container.keys())[-1]
        dummy_graph = global_container[dummy_split]['graphinfo'].graphify()
        for name in global_all_nodes:
            dummy_graph[name].x = encoder[name](graph_embeddings[dummy_split][name])
        if gnn_arch == "graphconv":
            graphnn = GeneralHeteroGraphConv(
                graph_for_dry_run = dummy_graph,
                hidden_dims = gnn_hidden_dims,
                out_dim = gnn_output_embed_dim,
                num_layers = gnn_num_layers,
                dropout = gnn_p_dropout,
                nonlin_fn = gnn_nonlin_fn,
                aggr_method=aggr_method,
                hetero_aggr_method=hetero_aggr_method,
                norm_type = norm_type,
                use_residual = use_residual,
            ).to(device)
        elif gnn_arch == "gat":
            graphnn = GeneralHeteroGAT(
                graph_for_dry_run = dummy_graph,
                hidden_dims = gnn_hidden_dims,
                out_dim = gnn_output_embed_dim,
                num_layers = gnn_num_layers,
                dropout = gnn_p_dropout,
                heads = gnn_n_heads,
                concat = concat,
                attn_dropout = attn_dropout,
                negative_slope = negative_slope,
                nonlin_fn = gnn_nonlin_fn,
                aggr_method=aggr_method,
                hetero_aggr_method=hetero_aggr_method,
                norm_type = norm_type,
                use_residual = use_residual,
            ).to(device)
        elif gnn_arch == "sage":
            graphnn = GeneralHeteroSAGE(
                graph_for_dry_run = dummy_graph,
                hidden_dims = gnn_hidden_dims,
                out_dim = gnn_output_embed_dim,
                num_layers = gnn_num_layers,
                dropout = gnn_p_dropout,
                nonlin_fn = gnn_nonlin_fn,
                aggr_method=aggr_method,
                hetero_aggr_method=hetero_aggr_method,
                norm_type = norm_type,
                use_residual = use_residual,
            ).to(device)
        elif gnn_arch == "rgcn":
            graphnn = GeneralHeteroRGCNConv(
                graph_for_dry_run = dummy_graph,
                hidden_dims = gnn_hidden_dims,
                out_dim = gnn_output_embed_dim,
                num_layers = gnn_num_layers,
                dropout = gnn_p_dropout,
                nonlin_fn = gnn_nonlin_fn,
                aggr_method=aggr_method,
                hetero_aggr_method=hetero_aggr_method,
                norm_type = norm_type,
                use_residual = use_residual,
            ).to(device)
        else:
            raise NotImplementedError()
        del dummy_graph

    for name in global_all_nodes:
        encoder[name] = encoder[name].to(device)
    
    print("--> Creating decoders ...")
    decoder: Dict[Tuple[str, str], nn.Module] = {}
    for _type in decoder_types:
        _cfg = cfg_decoder[_type]
        if _cfg['decoder_arch'] == "bilinear" or _cfg['decoder_arch'] == "dot_product":
            decoder[_type] = BilinearDecoder(
                embed_dims=_cfg['embed_dim'],
                is_identity=_cfg['decoder_arch'] == "dot_product",
                learnable_temp=_cfg['learnable_temp'],
            ).to(device)
        elif _cfg['decoder_arch'] == "mlp":
            decoder[_type] = MLPDecoder(
                embed_dims=_cfg['embed_dim'],
                mlp_configs=_cfg,
            ).to(device)

    print("--> Creating optimizers and schedulers...")
    optim: Dict[str, torch.optim.Optimizer] = {'graphnn': None, 'encoder': {}, 'decoder': {}}
    sched: Dict[str, Any] = {'graphnn': None, 'encoder': {}, 'decoder': {}}
    optim['graphnn'] = torch.optim.AdamW(
        graphnn.parameters(),
        lr=cfg_train.learning_rate['graphnn'],
        weight_decay=cfg_train.weight_decay['graphnn'],
    )
    for name in encoder.keys():
        if encoder[name] is not None and list(encoder[name].parameters()):
            optim['encoder'][name] = torch.optim.AdamW(
                encoder[name].parameters(),
                lr=cfg_train.learning_rate['encoder'][name],
                weight_decay=cfg_train.weight_decay['encoder'][name],
            )
    for name in decoder.keys():
        optim['decoder'][name] = torch.optim.AdamW(
            decoder[name].parameters(),
            lr=cfg_train.learning_rate['decoder'][name],
            weight_decay=cfg_train.weight_decay['decoder'][name],
        )
    n_steps = sum(1 for k in global_container.keys() if k.startswith('train_'))
    total_steps = cfg.training.n_epochs * n_steps
    for _name, _module in optim.items():
        if isinstance(_module, dict):
            for __name, __module in _module.items():
                sched[_name][__name] = getattr(
                    torch.optim.lr_scheduler, cfg_train.scheduler[_name][__name]
                )(optimizer=__module, T_max=total_steps)
        else:
            sched[_name] = getattr(
                torch.optim.lr_scheduler, cfg_train.scheduler[_name]
            )(optimizer=_module, T_max=total_steps)
    print("--> Optimizers and schedulers created.")

    if bundle is not None:
        moved = move_prep_to_device({
            "graph_embeddings": graph_embeddings,
            "qo_idx": qo_idx,
            "src_by_q": src_by_q,
            "dst_by_q": dst_by_q,
            "graphs": graphs,
        }, device)
        graph_embeddings = moved["graph_embeddings"]
        qo_idx = moved["qo_idx"]
        src_by_q = moved["src_by_q"]
        dst_by_q = moved["dst_by_q"]
        graphs = moved["graphs"]
    
    #################################################
    ##### Training Loop #############################
    #################################################
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.training.n_epochs):
        graphnn.train()
        for _name in encoder.keys(): encoder[_name].train()
        for _name in decoder.keys(): decoder[_name].train()

        _all_zero_grad(optim)
        for idx, curr_step in enumerate(torch.randperm(n_steps)):
            step = curr_step.item()
            loss, _, _, _ = run_forward_pass(
                embeddings=graph_embeddings[f'train_{step}'],
                encoders=encoder,
                graph=graphs[f'train_{step}'].to(device),
                graphnn=graphnn,
                link_pred_module=decoder,
                epoch=epoch,
                split_name=f'train_step{idx}',
                ce_pack=ce_packs[f'train_{step}'],
                device=device,
                loss_weighting=loss_w[f'train_{step}'],
                reduction=cfg_train.reduction,
                label_smoothing_alpha=cfg_train.label_smoothing_alpha,
            )
            (loss / cfg_train.accum_steps).backward()
            if (idx + 1) % cfg_train.accum_steps == 0 or (idx + 1) == n_steps:
                all_params = list(chain(
                    graphnn.parameters(),
                    *(m.parameters() for m in encoder.values() if m is not None),
                    *(m.parameters() for m in decoder.values() if m is not None),
                ))
                nn.utils.clip_grad_norm_(all_params,
                                         cfg_train.max_grad_norm,
                                         norm_type=cfg_train.grad_norm_type)
                _optim_all_step(optim)
                _sched_all_step(sched)
                _all_zero_grad(optim)
        
        if not ((epoch + 1) % cfg_train.eval_every == 0 or epoch == cfg.training.n_epochs - 1):
            continue

        graphnn.eval()
        for _name in encoder.keys(): encoder[_name].eval()
        for _name in decoder.keys(): decoder[_name].eval()

        with torch.no_grad():
            wave_prefixes = sorted(set(
                k.rsplit('_')[0] for k in global_container.keys() if k.endswith('_val')
            ))
            val_loss_list, val_acc_list, val_total_cnt_list = [], [], []
            test_loss_list, test_acc_list, test_total_cnt_list = [], [], []
            outs_for_saving: Dict[str, Dict[str, torch.Tensor]] = {}

            for wpref in wave_prefixes:
                loss_val, out_val, correct_cnt_val, total_cnt_val = run_forward_pass(
                    embeddings=graph_embeddings[f'{wpref}_val'],
                    encoders=encoder,
                    graph=graphs[f'{wpref}_val'].to(device),
                    graphnn=graphnn,
                    link_pred_module=decoder,
                    epoch=epoch,
                    split_name=f'{wpref}_val',
                    ce_pack=ce_packs[f'{wpref}_val'],
                    device=device,
                    loss_weighting=loss_w[f'{wpref}_val'],
                    reduction=cfg_train.reduction,
                    label_smoothing_alpha=1.0,
                )
                loss_test, out_test, correct_cnt_test, total_cnt_test = run_forward_pass(
                    embeddings=graph_embeddings[f'{wpref}_test'],
                    encoders=encoder,
                    graph=graphs[f'{wpref}_test'].to(device),
                    graphnn=graphnn,
                    link_pred_module=decoder,
                    epoch=epoch,
                    split_name=f'{wpref}_test',
                    ce_pack=ce_packs[f'{wpref}_test'],
                    device=device,
                    loss_weighting=loss_w[f'{wpref}_test'],
                    reduction=cfg_train.reduction,
                    label_smoothing_alpha=1.0,
                )
                val_loss_list.append(loss_val)
                test_loss_list.append(loss_test)
                val_acc_list.append(float(correct_cnt_val / total_cnt_val))
                test_acc_list.append(float(correct_cnt_test / total_cnt_test))
                val_total_cnt_list.append(total_cnt_val)
                test_total_cnt_list.append(total_cnt_test)
                outs_for_saving[wpref] = out_val # gnn output node embeddings for save

        loss_val = (
            torch.sum(torch.tensor(val_loss_list) * torch.tensor(val_total_cnt_list))
            / torch.sum(torch.tensor(val_total_cnt_list))
        ).detach().cpu()
        acc_val = (
            torch.sum(torch.tensor(val_acc_list) * torch.tensor(val_total_cnt_list))
            / torch.sum(torch.tensor(val_total_cnt_list))
        ).detach().cpu()
        loss_test = (
            torch.sum(torch.tensor(test_loss_list) * torch.tensor(test_total_cnt_list))
            / torch.sum(torch.tensor(test_total_cnt_list))
        ).detach().cpu()
        acc_test = (
            torch.sum(torch.tensor(test_acc_list) * torch.tensor(test_total_cnt_list))
            / torch.sum(torch.tensor(test_total_cnt_list))
        ).detach().cpu()
        print(f"Epoch {epoch:03d} | validation_loss: {loss_val:.4f}")
        print(f"Epoch {epoch:03d} | validation_acc: {acc_val:.4f}")
        print(f"Epoch {epoch:03d} | test_loss: {loss_test:.4f}")
        print(f"Epoch {epoch:03d} | test_acc: {acc_test:.4f}")

        loss_criterion = 1.0 - acc_val if cfg_exp.save_criterion == "acc" else loss_val
        if loss_criterion < best_val_loss:
            best_val_loss, patience_counter = loss_criterion, 0
            if cfg_exp.save_model:
                if cfg_exp.verbose:
                    print("Best validation loss, saving model...")
                torch.save({
                    'current_epoch': epoch,
                    'max_epochs': cfg.training.n_epochs,
                    'gnn_state_dict': graphnn.state_dict(),
                    'decoder_state_dict': {name: decoder[name].state_dict() for name in decoder.keys()},
                    'encoder_state_dict': {name: encoder[name].state_dict() for name in encoder.keys()},
                }, save_path_model)
            if cfg_exp.save_embedding:
                if cfg_exp.verbose:
                    print("Best validation loss, saving embeddings...")
                _save_dict = {
                    'label_to_x': global_label_to_x,
                    'per_wave': {}
                }
                for wpref in wave_prefixes:
                    _save_dict['per_wave'][wpref] = {'pre_gnn': {}, 'post_gnn': {}}
                    # save both GNN input and output node embeddings
                    for name in global_all_nodes.keys():
                        # GNN input embeddings live on the graph object
                        _save_dict['per_wave'][wpref]['pre_gnn'][name] = {
                            graph_labels[f'{wpref}_val'][name][i]: graphs[f'{wpref}_val'][name].x[i]
                            for i in range(len(graph_labels[f'{wpref}_val'][name]))
                        }
                        # GNN output embeddings come from outs_for_saving
                        _save_dict['per_wave'][wpref]['post_gnn'][name] = {
                            graph_labels[f'{wpref}_val'][name][i]: outs_for_saving[wpref][name][i]
                            for i in range(len(graph_labels[f'{wpref}_val'][name]))
                        }
                torch.save(_save_dict, save_path_embed)
                if cfg_exp.verbose:
                    print("Saved model and embeddings.")
        else:
            patience_counter += 1
            if patience_counter >= cfg_train.patience:
                if cfg_exp.verbose:
                    print("[STOP] Early stopping triggered.")
                break

if __name__ == "__main__":
    main()