from typing import Optional, Literal, List, Dict, Union
import random, os, json, pathlib

import torch
import numpy as np
from tqdm import tqdm

from sibyl.utils.surveydata_utils import (
    RawDataManagerATP,
    RawDataManagerHF,
)
from sibyl.constants.string_registry_survey import *
from sibyl.graph.entity import Individual, Question, Subgroup

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"


class NodeCollection:

    def __init__(self,
                 wave_number: Optional[List[int]] = None,
                 source: Literal['ATP', 'Twin', 'Eedi', 'dunning_kruger'] = 'ATP',
                 basepath: Union[str, pathlib.Path] = DATA_DIR / "opinionqa",
                 qkeypath: Union[str, pathlib.Path] = "opinionqa_question_keys.json",
                 qstringpath: Union[str, pathlib.Path] = "opinionqa_question_strings.json",
                 valid_traits: Optional[List[str]] = None,
                 use_selective_subgroups: bool = False,
                 min_questions_per_indiv: int = 30,
                 n_indiv_sample_per_wave: Optional[int] = None,
                 make_subgroup_nodes: bool = True,
                 **kwargs,
    ):
        """
        Args:
            wave_number (list):
                list of wave numbers to include in the graph formulation.
                some data for human simulation, especially surveys, comes in multiple waves
                (e.g., American Trends Panel Wave 26, Wave 81, ...)
                and the graph is built one per each wave.
                if specified instead of default (None), only the specified waves will be included.
            source (str):
                source of the human simulation data.
                'ATP' for American Trends Panel (OpinionQA),
                'Twin' for the Twin-2k-500 dataset,
                'Eedi' for the Eedi Neurips 2020 education challenge dataset,
                'dunning_kruger' for the Dunning-Kruger effect dataset (https://www.nature.com/articles/s41562-021-01057-0).
            basepath (str, path):
                path to the dataset files. Default points to the opinionqa data folder.
            qkeypath (str, path):
                path to question key file that contains the list of question keys.
            qstringpath (str, path):
                path to question string file that contains the mapping from question keys to question strings.
            valid_traits (list):
                list of demographic traits to consider for subgroup creation.
                Default uses all nine traits defined in VALID_TRAITS. Change this to adapt to your needs.
            use_selective_subgroups (bool):
                if True, only create subgroups that are defined in INTERESTED_GROUPS.
                for example, even if 'age' is a valid trait, you can create only 'age_18-29' and 'age_65+' subgroups
                by configuring INTERESTED_GROUPS accordingly.
                But in our experiments we use all possible subgroups.
            min_questions_per_indiv (int):
                minimum number of questions answered per individual to be included in the graph.
                individuals with less than this number of answered questions will be filtered out.
            n_indiv_sample_per_wave (int):
                if specified, randomly sample this number of individuals per wave (per graph).
                it is useful for debugging with a smaller graph, but not used for actual training.
            make_subgroup_nodes (bool):
                if True (default), create subgroup nodes in the graph.
                if False, only individual and choice nodes are created, resulting in a standard bipartite graph.
        """
        self.wave_number: Optional[List[int]] = wave_number
        self.source: str = source
        self.individuals: Dict[float, Individual] = {}
        self.subgroups: Dict[str, Subgroup] = {}
        self.valid_traits: List[str] = (
            valid_traits if valid_traits is not None
            else VALID_TRAITS
        )

        # we provide a dataset-specific handler, due to significant differences in data format
        # please refer to sibyl/utils/surveydata_utils.py and the docstrings below for details.
        if source == 'ATP':
            self.manager: RawDataManagerATP = RawDataManagerATP(
                qkeypath=qkeypath,
                qstringpath=qstringpath,
                datapath=basepath,
                selected_waves=wave_number,
            )
            self.qkeys: List[str] = [
                qkey for qkey in self.manager.qkeys
                if wave_number is None or int(qkey.split(DELIMITER)[-1]) in wave_number
            ]
            self.survey_waves: List[int] = [
                wave for wave in self.manager.survey_waves
                if wave_number is None or wave in wave_number
            ]
            self.qkeys.sort(); self.survey_waves.sort()
            self.options_map: Dict[str, Dict] = {
                qkey: self.manager.fetch_options(qkey=qkey)
                for qkey in self.qkeys
            }
            self.q_map: Dict[str, str] = self.manager.qstrings

        if source == 'Twin' or source == 'dunning_kruger':
            dataset_name = kwargs.get('dataset_name', None)
            assert dataset_name is not None, "dataset_name required for HuggingFace datasets."
            self.manager: RawDataManagerHF = RawDataManagerHF(
                qkeypath=qkeypath,
                qstringpath=qstringpath,
                datapath=basepath,
                rawdatapath=f"{dataset_name}_responses.csv",
                options_map_path=f"{dataset_name}_options_map.json",
            )
            with open(os.path.join(basepath, qkeypath), 'r') as f:
                self.qkeys: List[str] = json.load(f)
                self.qkeys.sort()
            self.survey_waves: List[int] = [1]
            with open(os.path.join(basepath, f"{dataset_name}_options_map.json"), 'r') as f:
                self.options_map: Dict[str, Dict] = json.load(f)
                for q, opts in self.options_map.items():
                    updates = {}
                    for k, v in opts.items():
                        k_float = float(k)
                        updates[k_float] = v
                    self.options_map[q] = updates
            with open(os.path.join(basepath, qstringpath), 'r') as f:
                self.q_map: Dict[str, str] = json.load(f)

        if source == 'Eedi':
            raise NotImplementedError()
        
        # create individual nodes at self.individuals
        self._load_individuals(wave_number=wave_number,
                               minq=min_questions_per_indiv,
                               n_indiv_sample=n_indiv_sample_per_wave)
        # create subgroup nodes at self.subgroups if make_subgroup_nodes is True
        if make_subgroup_nodes:
            self._make_subgroups(use_selective_subgroups=use_selective_subgroups)
        
    def _load_individuals(self,
                          wave_number: Optional[List[int]] = None,
                          minq: int = 30, # minimum questions per individual
                          n_indiv_sample: Optional[int] = None, # number of individuals to sample
                          seed: int = 42, # seed for random sampling of individuals
        ) -> None:
        """
        Semantics:
            load individuals from the raw data manager and populate self.individuals.
            each individual is represented as an Individual object.
            filter out individuals with less than minq questions answered;
            if n_indiv_sample is specified, randomly sample this number of individuals.
            each individual has information about their individual features
                - this is handled by self._update_individuals_identity()
            and their responses to questions
                - this is handled by self._update_individuals_response()
        """
        # count the number of total / unique individuals
        uids_set: set[float] = set()
        n_count: int = 0
        for wave in self.survey_waves:
            respondent_ids: np.ndarray = self.manager.fetch_respondent(
                wave_number=wave,
            )
            uids_set.update(respondent_ids)
            n_count += respondent_ids.shape[0]
        self.ids: set[float] = uids_set
        print(f"_load_individuals(): count of individuals: {n_count}")
        print(f"_load_individuals(): unique ids: {len(self.ids)}")

        # add demographic info to individuals
        for trait in self.valid_traits:
            for wave in self.survey_waves:
                if wave_number is not None and wave not in wave_number:
                    continue
                self._update_individuals_identity(
                    trait=trait, wave=wave,
                )
        print(f"_load_individuals(): updated individual features for {len(self.valid_traits)} traits.")

        # add response info to individuals
        for qkey in tqdm(
            self.qkeys,
            desc="Updating response info"
        ):
            if wave_number is not None:
                curr_wave = int(qkey.split(DELIMITER)[-1])
                if curr_wave not in wave_number:
                    continue
            self._update_individuals_response(qkey=qkey)
        print("_load_individuals(): updated response info of "
              f"{len(self.qkeys)} questions.")
        remove_cnt : int = 0
        total_cnt : int = len(self.individuals)
        for indiv_id in list(self.individuals.keys()):
            if len(self.individuals[indiv_id].respond_info) < minq:
                del self.individuals[indiv_id]
                remove_cnt += 1
        print(f"_load_individuals(): removed {remove_cnt} individuals"
              f" from {total_cnt} with less than {minq} questions answered.")
        random.seed(seed)
        if n_indiv_sample is not None:
            n_sample = min(n_indiv_sample, len(self.individuals))
            self.individuals = dict(
                random.sample(list(self.individuals.items()), n_sample)
            )
            print(f"_load_individuals(): randomly sampled "
                  f"{n_sample} individuals.")
        self.ids = set(self.individuals.keys())
        return


    def _update_individuals_response(self, qkey: str) -> None:
        """
        Semantics:
            for each qkey, load dataframe and update all indiv's response info.
        """
        wave = int(qkey.split(DELIMITER)[-1])
        df = self.manager.response[wave][
            ['QKEY', qkey, 'WEIGHT_W' + str(wave)]
        ]
        for row in df.itertuples():
            uid, value = row.QKEY, getattr(row, qkey)
            uid = uid * 10000.0 + wave
            if value not in self.options_map[qkey]:
                continue
            if uid not in self.individuals:
                self.individuals[uid] = Individual(id=uid, source=self.source)
            self.individuals[uid].update_respond_info(
                key=qkey, value=float(value)
            )
            self.individuals[uid].update_weight(
                wave=wave, weight=getattr(row, 'WEIGHT_W' + str(wave))
            )
        return
    
    def _update_individuals_identity(self,
                                     trait: str,
                                     wave: int) -> None:
        """
        Semantics:
            for each trait and wave number, search for the trait column in response.
            Then map the trait name to the string.
            Update the identity of each individual in dataframe.
        Important notes:
            Each human simulation dataset has different ways of storing individual features.
            For example, in OpinionQA (American Trends Panel),
            the features are stored in the Dataframe columns with numeric values,
            and the mapping from numeric values to string values (e.g., 1.0 to 'male') are stored in metadata;
            such numeric-to-string mapping is different for each wave
            (e.g., in one wave, 1.0 means 'income $10-20k' and 2.0 means 'income $20-30k',
            but in another wave, 1.0 means 'income $10-30k').
            Another example is the Twin-2k-500 dataset,
            where the individual features are already stored in the Dictionary format not in Dataframe.
            We did our best to accommodate these differences in the mapping process by ad-hoc handling.
        """
        column_to_search = globals().get(f"{self.source}_{trait.upper()}_COLUMN", None)
        assert column_to_search is not None, (
            f"{trait} not recognized for source {self.source}."
        )
        for column in column_to_search:
            # continue to the next possible column if the current candidate column is not found
            if column not in self.manager.response[wave].columns:
                continue
            
            if self.source == 'ATP':
                number_to_str_mapping = self.manager.meta[
                    wave
                ].variable_value_labels[column]
                str_to_str_mapping = globals().get(
                    f"{self.source.upper()}_CONVERSION_TABLE", {}
                )[column.lower()]
                df = self.manager.response[wave][['QKEY', column]]
                for row in df.itertuples():
                    uid, value = row.QKEY, getattr(row, column)
                    uid = uid * 10000.0 + wave
                    if uid not in self.individuals:
                        self.individuals[uid] = Individual(id=uid, source=self.source)
                    value_str = number_to_str_mapping.get(
                        value, ""
                    ).lower().split('(')[0].strip()
                    value_final = str_to_str_mapping.get(value_str, "")
                    if value_final == "":
                        if all(flag not in value_str for flag in INVALID_FLAGS) \
                            and not np.isnan(value):
                            raise ValueError(
                                f"Value '{value}' ({value_str}) not found in "
                                f"mapping for {trait} in wave {wave} {column}."
                            )
                        continue
                    _ = self.individuals[uid].identity.update_trait(
                        trait=trait, value=value_final
                    )
                return

            if self.source == 'Twin' or self.source == 'dunning_kruger':
                df = self.manager.response[wave][['QKEY', column]]
                for row in df.itertuples():
                    uid, value = row.QKEY, getattr(row, column)
                    uid = uid * 10000.0 + wave
                    if uid not in self.individuals:
                        self.individuals[uid] = Individual(id=uid, source=self.source)
                    value_final = self.options_map[column][float(value)].lower()
                    _ = self.individuals[uid].identity.update_trait(
                        trait=trait, value=value_final
                    )
                return
            
            if self.source == 'Eedi':
                raise NotImplementedError()
            
        # failure to locate the individual feature information, should not happen
        assert False, f"Trait {trait} not found in wave {wave} response."

    def _make_subgroups(self, use_selective_subgroups: bool = False) -> None:
        """
        Semantics:
            create subgroups based on self.valid_traits and self.individuals,
            and populate self.subgroups.
            self.subgroups is a dictionary that maps subgroup keys to Subgroup objects,
            where Subgroup objects contain the list of individual ids that belong to the subgroup.
            Currently supports only single-trait subgroups (e.g., 'age_18-29')
            but Subgroup class can handle multi-trait subgroups (e.g., age and gender) during initialization.
        """
        _cnt = 0
        for trait in self.valid_traits: # e.g., 'age'
            for attr in INDIV_FEAT_ENCODING[self.source][trait]: # e.g., '18-29', '30-49', ...
                subgroup_key = f"{trait}_{attr}" # format of subgroup key: 'age_18-29'
                if use_selective_subgroups and subgroup_key not in INTERESTED_GROUPS:
                    continue
                _cnt += 1
                self.subgroups[subgroup_key] = Subgroup(
                    traits=[trait],
                    attributes=[attr],
                    init_individuals=self.individuals,
                    source=self.source,
                )
        print(f"_make_subgroups(): created {_cnt} subgroups.")
        return

    def __iter__(self):
        for indiv in self.individuals.values():
            yield indiv

    def __len__(self):
        return len(self.individuals)
    
    def nodify_indiv(self,
                     embedding_scheme,
                     add_self_loops: bool,
                     **kwargs,
                     ) -> List[Dict]:
        return [
            indiv.nodify(
                add_self_loops=add_self_loops,
                valid_traits=self.valid_traits,
                exit_undefined=kwargs.get('exit_undefined', False),
                unique_idx=idx,
                subgroups=list(self.subgroups.values()),
            ) for idx, indiv in enumerate(self.individuals.values())
        ]
    
    def nodify_subgroup(self,
                        embedding_scheme: Literal[
                            'one_hot',
                            'fixed'
                        ],
                        add_self_loops: bool,
                        **kwargs,
                        ) -> List[Dict]:
        assert embedding_scheme in ['one_hot', 'fixed'], (
            f"Invalid embedding scheme for subgroups: {embedding_scheme}"
        )
        return_nodes: List[Dict] = []
        for subgroup in self.subgroups.values():
            node = subgroup.nodify(
                embedding_scheme=embedding_scheme,
                add_self_loops=add_self_loops,
            )
            return_nodes.append(node)
        return return_nodes
    
    def nodify_question(self,
                        embedding_scheme: Literal[
                            'one_hot',
                            'random',
                            'frozen_llm_projection'
                        ],
                        add_self_loops: bool,
                        **kwargs,
                        ) -> List[Dict]:
        return_nodes: List[Dict] = []
        start_idx: int = 0
        embedding_dict: Dict[str, torch.Tensor] = {}
        if embedding_scheme.startswith('frozen_llm_projection'):
            embedding_dict = kwargs.get('embedding_dict', None)
            assert embedding_dict is not None, (
                "LLM embedding must be provided for frozen_llm_projection."
            )
        for qkey in self.qkeys:
            q: Question = Question(
                qkey=qkey,
                qstring=self.q_map[qkey],
                options=self.options_map[qkey],
            )
            nodes = q.nodify(
                embedding_scheme=embedding_scheme,
                add_self_loops=add_self_loops,
                embedding_dict=embedding_dict,
                start_idx=start_idx,
            )
            return_nodes.extend(nodes)
            start_idx += len(q.options)
        return return_nodes


def nodify_all(collection: NodeCollection,
               embedding_scheme: Dict[str, str],
               add_self_loops: bool,
               node_types: List[str],
               **kwargs,
               ) -> Dict[str, List[Dict]]:
    """
    Semantics:
        given a NodeCollection object, create nodes for the specified node_types
        (at most 'indiv', 'subgroup', 'question'),
        with the embedding_scheme per node_type and add_self_loops option.
        return a dictionary that maps node type to the list of node dictionaries.
    """
    all_nodes = {
        name: getattr(collection, f"nodify_{name}")(
            embedding_scheme=embedding_scheme[name],
            add_self_loops=add_self_loops,
            **kwargs,
        ) for name in node_types
    }
    for name, nodes in all_nodes.items():
        all_nodes[name] = [node for node in nodes if node is not None]
    return all_nodes


def map_label_to_x(nodes: List[Dict]):
    return {node['label']: node['x'] for node in nodes}