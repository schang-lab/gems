from typing import Dict, Optional, Any, Literal, List, Tuple

import torch
import numpy as np

from sibyl.constants.string_registry_survey import (
    VALID_TRAITS,
    DELIMITER,
    INDIV_FEAT_ENCODING,
    PREDEFINED_ENCODING,
)

class Identity:

    def __init__(self):
        self.attributes: Dict[str, str] = {}

    def update_trait(self,
                     trait: str,
                     value: str) -> Optional[str]:
        """
        Semantics: Update the trait of the identity.
        If the trait exists and the value is different, return existing one.
        """
        assert isinstance(trait, str) and trait in VALID_TRAITS
        existing_value: Optional[str] = None
        if trait in self.attributes:
            existing_value = self.attributes[trait]
        self.attributes[trait] = value
        return existing_value if existing_value != value else None
    
    def get_trait(self, trait: str) -> Optional[str]:
        if trait in self.attributes:
            return self.attributes[trait]
        return None

    def __repr__(self) -> str:
        repr_str = "Identity("
        repr_str += ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        repr_str += ")"
        return repr_str
    
    def __eq__(self, other: 'Identity') -> bool:
        return isinstance(other, Identity) and \
            self.attributes == other.attributes


class Individual:
    
    def __init__(self, id: float, source: str):
        self.id: float = id
        self.weight: Dict[int, float] = {}
        # key indicates wave number, value indicates weight
        self.identity: Identity = Identity()
        self.respond_info: Dict[str, float] = {}
        self.participated: Dict[int, bool] = {}
        self.source: str = source

    def update_weight(self, wave: int, weight: float) -> None:
        assert isinstance(wave, int)
        self.weight[wave] = weight if not np.isnan(weight) else 0.0

    def update_respond_info(self,
                            key: str,
                            value: float) -> None:
        """
        Semantics: Update the response info of the individual.
        If the key already exists, raise an error.
        Also update the participation info based on the wave.
        (Refer to update_participation method.)
        """
        if not (isinstance(key, str) and isinstance(value, float)):
            import pdb; pdb.set_trace()
        assert key not in self.respond_info, \
            f"Key {key} already exists in respond_info of {self.id}"
        self.respond_info[key] = value
        wave = int(key.split(DELIMITER)[-1])
        self.update_participation(wave=wave, pcp=True)

    def update_participation(self,
                             wave: int,
                             pcp: bool) -> None:
        assert isinstance(wave, int) and isinstance(pcp, bool)
        if wave in self.participated:
            assert self.participated[wave] == pcp, (
                f"Incosistent participation info: {self.id}, {wave}"
            )
        self.participated[wave] = pcp

    def merge_info(self, other: 'Individual') -> None:
        assert self == other, \
            f"Cannot merge different indiv.: {self.id} + {other.id}"
        new = self.__class__.__new__(self.__class__)
        new.id = self.id
        new.identity = self.identity
        new.respond_info = self.respond_info.copy()
        for k, v in other.respond_info.items():
            self.update_respond_info(k, v)
        self.participated.update(other.participated)
        return new
    
    def nodify(self,
               add_self_loops: bool,
               valid_traits: List[str],
               exit_undefined: bool,
               unique_idx: int,
               subgroups: Optional[List["Subgroup"]] = None,
               ) -> Optional[Dict[str, Any]]:
        package: Dict[str, Any] = {}
        # input embeddings
        feats: List[torch.Tensor] = []
        for trait in valid_traits:
            one_idx = INDIV_FEAT_ENCODING[self.source][trait].get(
                self.identity.attributes.get(trait, ""), -1
            )
            if one_idx == -1 and exit_undefined:
                return None
            feats.append(one_idx)
        feats.append(unique_idx)
        # edges (weights default to 1.0)
        self_label = str(round(self.id))
        edges_dict: Dict[
            Tuple[str, str, str],
            List[Tuple[str, float]]
        ] = {
            ('indiv', 'responds', 'question'): [],
            ('indiv', 'self', 'indiv'): [],
        }
        for qkey, option in self.respond_info.items():
            node_to = qkey + "_option_" + str(round(option))
            edges_dict[('indiv', 'responds', 'question')].append(
                (node_to, 1.0)
            )
        if subgroups:
            edges_dict[('indiv', 'contains', 'subgroup')] = []
            for sg in subgroups:
                if all(self.identity.get_trait(t) == a for t, a in sg.info):
                    edges_dict[('indiv', 'contains', 'subgroup')].append((
                        "+".join([
                            f"{trait}_{attr}" for trait, attr in sg.info
                        ]), 1.0
                    ))
        if add_self_loops:
            edges_dict[('indiv', 'self', 'indiv')].append(
                (self_label, 1.0)
            )

        package['node_type'] = 'indiv'
        package['label'] = self_label
        package['x'] = feats
        package['edges'] = edges_dict
        return package

    def __eq__(self, other: 'Individual') -> bool:
        return isinstance(other, Individual) \
            and self.id == other.id \
            and self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.id)
        
    def __repr__(self) -> str:
        repr_str = f"Individual(id={self.id}, "
        repr_str += f"participated={self.participated}, "
        repr_str += f"respond_info={self.respond_info}), "
        repr_str += f"identity={repr(self.identity)})"
        return repr_str


class Subgroup:
    
    def __init__(
        self,
        traits: List[str],
        attributes: List[str],
        source: str,
        init_individuals: Dict[float, Individual] = {},
    ):
        assert len(traits) == len(attributes) and len(traits) > 0
        self.info: List[Tuple[str, str]] = list(zip(traits, attributes))
        self.individual_ids: set[float] = set()
        for id, indiv in init_individuals.items():
            identity = indiv.identity
            success: bool = True
            for trait, attr in self.info:
                if not identity.get_trait(trait) == attr:
                    success = False
                    break
            if success:
                self.individual_ids.add(id)
        self.source: str = source
        return
    
    def nodify(self,
               embedding_scheme: Literal['one_hot', 'fixed'],
               add_self_loops: bool,
               ) -> Dict[str, Any]:
        assert embedding_scheme in ['one_hot', 'fixed'], (
            f"Invalid embedding scheme for subgroups: {embedding_scheme}"
        )
        package: Dict[str, Any] = {}
        package['node_type'] = 'subgroup'
        package['label'] = "+".join(
            [f"{trait}_{attr}" for trait, attr in self.info]
        )
        if embedding_scheme == 'fixed':
            package['x'] = torch.tensor(1.0, dtype=torch.float32)
        else:
            package['x'] = PREDEFINED_ENCODING[self.source].index(
                ", ".join(f"{trait}_{attr}" for trait, attr in self.info)
            )
        edges_dict: Dict[Tuple[str, str, str], List[Tuple[str, float]]] = {}
        if add_self_loops:
            edges_dict[('subgroup', 'self', 'subgroup')] = [
                (package['label'], 1.0)
            ]
        package['edges'] = edges_dict
        return package
    
    def __repr__(self) -> str:
        repr_str = "Subgroup("
        repr_str += ", ".join(
            f"{trait}={attr}" for trait, attr in self.info
        )
        repr_str += f", individuals={len(self.individual_ids)})"
        return repr_str
        

class Question:
    
    def __init__(self,
                 qkey: str,
                 qstring: str,
                 options: Dict[str, float]):
        self.qkey: str = qkey
        self.qstring: str = qstring
        self.options: Dict[str, float] = options

    def nodify(self,
               embedding_scheme: Literal['one_hot', 'random', 'frozen_llm_projection'],
               add_self_loops: bool,
               start_idx: Optional[int] = None,
               embedding_dict: Optional[torch.Tensor] = None,
               ) -> Dict[str, Any]:
        packages: List[Dict[str, Any]] = []

        options_list = list(self.options.items())
        for idx, (option, value) in enumerate(options_list):
            package: Dict[str, Any] = {}
            package['node_type'] = 'question'
            package['label'] = (
                self.qkey + "_option_" + str(round(option))
            )
            if embedding_scheme.startswith('frozen_llm_projection'):
                assert embedding_dict is not None
                package['x'] = embedding_dict[package['label']]
            else:
                assert start_idx is not None, (
                    "Idx. must be specified for one-hot or random encoding."
                )
                package['x'] = start_idx + idx
            if add_self_loops:
                package['edges'] = {
                    ('question', 'self', 'question'): [
                        (package['label'], 1.0)
                    ]
                }
            packages.append(package)

        return packages
