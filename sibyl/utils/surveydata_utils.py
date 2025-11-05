import os
import json
import pathlib
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union, final, Literal

import pandas as pd
import numpy as np
import pyreadstat

from sibyl.constants.string_registry_survey import INVALID_FLAGS
from sibyl.utils.misc_utils import (
    ordinal_emd,
    convert_for_json,
    normalized_mutual_information,
    normalized_entropy,
)
from sibyl.utils.stats import Distribution


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))


class RawDataManagerHF:
    """
    HuggingFace downloadable dataset manager.
    """
    def __init__(
        self,
        qkeypath: Union[str, pathlib.Path] = "twin_qkeys.json",
        qstringpath: Union[str, pathlib.Path] = "twin_qstrings.json",
        datapath: Union[str, pathlib.Path] = os.path.join(ROOT_DIR, "data", "twin"),
        rawdatapath: Union[str, pathlib.Path] = "twin_responses.csv",
        options_map_path: Union[str, pathlib.Path] = "twin_options_map.json",
    ):
        self.qkeypath = qkeypath
        self.qstringpath = qstringpath
        self.datapath = datapath

        self.survey_family: str = "twin"
        self.qkeys = json.load(open(os.path.join(datapath, qkeypath), "r"))
        self.qstrings = json.load(open(os.path.join(datapath, qstringpath), "r"))
        waves_list: List[int] = [1]
        self.survey_waves = waves_list
        self.response: Dict[int, pd.DataFrame] = {}
        self.response[1] = pd.read_csv(os.path.join(datapath, rawdatapath))
        with open(os.path.join(datapath, options_map_path), "r") as f:
            self.options_map = json.load(f)
            for k, v in self.options_map.items():
                self.options_map[k] = {float(key): val for key, val in v.items()}
        return

    def fetch_respondent(self,
                         wave_number: Optional[int] = None,
                         qkey: Optional[Union[str, List[str]]] = None
                         ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if qkey is None:
            return np.array(self.response[1].QKEY)

    def _check(self, condition, message):
        if not condition:
            if self.warning:
                warnings.warn(message)
            else:
                assert False, message

    def fetch_options(
        self, qkey: str, exclude_last: bool = False
    ) -> Tuple[Dict[Any, str], List[int]]:
        wave_number: int = 1
        assert wave_number in self.survey_waves, (
            f"--> RawDataManagerATP::fetch_options(): invalid wave {wave_number}."
        )
        self._check((isinstance(qkey, str)
                and qkey in self.qkeys
                and qkey in self.response[wave_number].columns), (
            f"--> RawDataManagerATP::fetch_options(): invalid qkey {qkey}."
        ))
        options = self.options_map.get(qkey, {})
        options = {
            value: label for value, label in options.items()
            if isinstance(value, float)
        }
        return options
    

class RawDataManager:
    
    def __init__(self, warning: bool = False):
        """
        Semantics:
            Base class for human simulation data managers.
            Provides basic interfaces, including fetching respondent ids, responses, options, and weights.
            Please refer to the method docstrings for details.
        Args:
            qkeypath, qstringpath, datapath, selected_waves: please refer to the docstring of NodeCollection.
            warning (bool):
                if True, raise warnings instead of errors when unexpected events happen.
                if False, raise errors (default).
        """
        self.survey_family: str = ""
        self.survey_waves: List[str] = []
        self.qkeys: List[str] = []
        self.warning: bool = False

    def __repr__(self) -> str:
        return_info = []
        return_info.append("=" * 20 + " Raw Data Manager " + "=" * 20)
        return_info.append(f"Survey Family: {self.survey_family}")
        return_info.append(f"Survey Waves: {self.survey_waves}")
        return_info.append(f"QKeys: {self.qkeys}")
        return_info.append("=" * 57 + "\n")
        return "\n".join(return_info)
    
    def _check(self, condition, message):
        if not condition:
            if self.warning:
                warnings.warn(message)
            else:
                assert False, message


@final
class RawDataManagerATP(RawDataManager):
    
    def __init__(
        self,
        qkeypath: Union[str, pathlib.Path],
        qstringpath: Union[str, pathlib.Path],
        datapath: Union[str, pathlib.Path],
        selected_waves: Optional[List[int]] = None,
        warning: bool = False,
    ):
        super().__init__(warning=warning)
        self.survey_family = "ATP"
        
        # loading data from .json files for question keys and strings
        self.qkeys: List[str] = json.load(
            open(os.path.join(datapath, qkeypath), "r")
        )
        self.qstrings: Dict[str, str] = json.load(
            open(os.path.join(datapath, qstringpath), "r")
        )
        waves_list: List[int] = [
            self._wave_number(qkey) for qkey in self.qkeys
        ]
        self.survey_waves: List[int] = sorted(list(set(waves_list)))

        # filtering survey waves and qkeys if selected_waves is provided
        if selected_waves is not None:
            self.survey_waves = [w for w in self.survey_waves if w in selected_waves]
            self.qkeys = [q for q in self.qkeys if self._wave_number(q) in selected_waves]

        # loading data from the .sav files (directly downloaded from Pew Research)
        # raw data is stored in self.response (DataFrame) and self.meta (MetaData)
        self.response: Dict[int, pd.DataFrame] = {}
        self.meta: Dict[int, pyreadstat._readstat_parser.metadata_container] = {}
        for wave in self.survey_waves:
            expected_path = os.path.join(datapath, "raw_data", f"ATP W{wave}.sav")
            assert os.path.exists(expected_path), (
                f"--> RawDataManagerATP: ATP W{wave}.sav does not exist."
            )
            self.response[wave], self.meta[wave] = pyreadstat.read_sav(expected_path)


    def fetch_options(
        self, qkey: str, exclude_last: bool = True
    ) -> Dict[float, str]:
        """
        Semantics:
            Fetch the options for a given question key (qkey) in a {'value': 'label'} format.
            For example, {1.0: 'Yes', 2.0: 'No'}.
            If exclude_last is True (default), exclude the last option.
            In ATP surveys, the last option is invalid;
            for more information, refer to INVALID_FLAGS at sibyl/constants/string_registry_survey.py
        """
        wave_number: int = self._wave_number(qkey)
        assert wave_number in self.survey_waves, (
            f"--> RawDataManagerATP::fetch_options(): invalid wave {wave_number}."
        )
        self._check((isinstance(qkey, str)
                and qkey in self.qkeys
                and qkey in self.response[wave_number].columns), (
            f"--> RawDataManagerATP::fetch_options(): invalid qkey {qkey}."
        ))
        options = self.meta[wave_number].variable_value_labels.get(qkey, {})
        options = deepcopy(options)
        exclude_value = list(options.keys())[-1]
        if exclude_last:
            exclude_label = options.pop(exclude_value)
            self._check(
                any(flag in exclude_label.lower() for flag in INVALID_FLAGS),
                (f"--> RawDataManagerATP::fetch_options(): "
                 f"last option {exclude_value} ({exclude_label}) may be valid.")
            )
        options = {
            value: label for value, label in options.items()
            if isinstance(value, float)
        }
        return options

    def _fetch_respondent(self,
                          wave_number: int,
                          qkey: Optional[str] = None
                          ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Semantics:
            when qkey is None, return ids of all respondents in the wave.
            when qkey is not None, return ids of respondents and their responses.
        """
        assert isinstance(wave_number, int) and wave_number in self.survey_waves, (
            f"--> RawDataManagerATP::_fetch_respondent(): invalid wave {wave_number}."
        )
        ids =  np.array(self.response[wave_number].QKEY)
        if qkey is None:
            return np.sort(ids)
        self._check((isinstance(qkey, str)
                and qkey in self.qkeys
                and qkey in self.response[wave_number].columns), (
            f"--> RawDataManagerATP::_fetch_respondent(): invalid qkey {qkey}."
        ))
        responses = np.array(getattr(self.response[wave_number], qkey))
        options = self.fetch_options(qkey)
        values = options.keys()
        mask = [not pd.isna(r) and r in values for r in responses]
        ids, responses = ids[mask], responses[mask]
        sort_idx = np.argsort(ids)
        return ids[sort_idx], responses[sort_idx]

    def fetch_respondent(self,
                         wave_number: Optional[int] = None,
                         qkey: Optional[Union[str, List[str]]] = None
                         ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Semantics:
            when qkey is None, return ids of all respondents in the wave.
            when qkey is not None,
            return ids and responses of respondents who answered all questions in qkey.
        """
        assert (wave_number is None and qkey is not None) \
            or (wave_number is not None and qkey is None), (
            "--> RawDataManagerATP::fetch_respondent(): "
            "exactly one of wave_number or qkey must be None."
        )
        if qkey is None:
            return self._fetch_respondent(wave_number=wave_number)
        if isinstance(qkey, str):
            qkey = [qkey]
        ids, responses = self._fetch_respondent(
            wave_number=self._wave_number(qkey[0]), qkey=qkey[0]
        )
        responses = responses.reshape(-1, 1)
        for q in qkey[1:]:
            ids_, responses_ = self._fetch_respondent(
                wave_number=self._wave_number(q), qkey=q
            )
            intersec = set(ids) & set(ids_)
            old_mask = [i in intersec for i in ids]
            new_mask = [i in intersec for i in ids_]
            ids, ids_ = ids[old_mask], ids_[new_mask]
            responses = np.hstack((
                responses[old_mask,:],
                responses_.reshape(-1, 1)[new_mask,:]
            ))
        if ids.shape[0] == 0:
            print(
                "--> RawDataManagerATP::fetch_respondent(): warning: "
                f"qkeys {qkey} have no common responses."
            )
        return ids, responses
    
    def fetch_weights(self,
                      wave_number: int,
                      target_ids: Optional[List[int]] = None
    ) -> Union[Dict[int, float], np.ndarray]:
        """
        Semantics:
            Fetch respondent weights for a given wave number.
            If target_ids is provided, only fetch weights for those ids.
        """
        ids = np.array(self.response[wave_number].QKEY)
        weights = np.array(
            getattr(self.response[wave_number], f"WEIGHT_W{wave_number}")
        )
        mask = [not pd.isna(w) for w in weights]
        id_weight_pair = {
            id: weight
            for id, weight in zip(ids[mask], weights[mask])
        }
        if target_ids is None:
            return id_weight_pair
        return np.array([id_weight_pair[id] for id in target_ids])    
    

    @staticmethod
    def _wave_number(qkey: str) -> int:
        assert isinstance(qkey, str)
        return int(qkey.split("_W")[-1])


class SurveyQuestion:

    def __init__(self, sample):
        self.survey_family: str = sample["survey_family"]
        self.qkey: str = sample["qkey"]
        self.attribute: str = sample["attribute"]
        self.group: str = sample["group"]
        self.question: str = sample["question"].strip()
        self.options: List[str] = [opt.strip() for opt in sample["options"]]
        self.responses: List[float] = sample["responses"]
        self.refusal_rate: float = sample["refusal_rate"]
        self.ordinal: List[float] = sample["ordinal"]
        self.survey_wave: str = self.get_survey_wave()
        
        assert len(self.options) == len(self.responses) + 1, (
            f"--> SurveyQuestion: for question {self.qkey}, "
            f"number of options (including refusal) is {len(self.options)}, "
            f"but number of responses and refusal is {len(self.responses) + 1}."
        )
        assert len(self.responses) == len(self.ordinal), (
            f"--> SurveyQuestion: for question {self.qkey}, "
            f"number of responses is {len(self.responses)}, "
            f"but number of ordinality information is {len(self.ordinal)}."
        )

    def get_median_month_year(self) -> str:
        start_str, end_str = self.survey_conducted_date.split("_")
        start_date = datetime.strptime(start_str, "%Y%m%d")
        end_date = datetime.strptime(end_str, "%Y%m%d")
        delta = end_date - start_date
        median_date = start_date + timedelta(days=delta.days // 2)
        return median_date.strftime("%B %Y")

    def get_survey_wave(self) -> str:
        if self.survey_family == 'ATP':
            return self.qkey.split('_W')[-1]
        elif self.survey_family == 'GSS':
            return self.qkey.split('_year_')[-1]
        raise ValueError(
            f"--> SurveyQuestion: unknown survey family {self.survey_family}."
        )

    def convert_temporal_information(self, year_only:bool = True) -> str:
        return (
            ""
            + (
                self.get_median_month_year() if not year_only
                else "year " + self.get_median_month_year().split(" ")[-1].strip()
            )
            + ".\n"
        )

    def get_emd(self, estimation: List[float]) -> float:
        assert len(estimation) == len(self.responses), (
            f"--> SurveyQuestion: estimation and responses should be same length."
            f" Length: estimation = {len(estimation)}, responses = {len(self.responses)}."
            f" Check whether you accidentally included refusal probability."
        )
        return ordinal_emd(self.responses, estimation, self.ordinal)

    def to_dict(self) -> Dict[str, Any]:
        return {k:v for k,v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self) -> str:
        return_info = []
        return_info.append("=" * 20 + " Survey Question " + "=" * 20)
        return_info.append(f"Survey Family: {self.survey_family}")
        return_info.append(f"Survey Wave: {self.survey_wave}")
        return_info.append(f"Survey Conducted Date: {self.survey_conducted_date}")
        return_info.append(f"QKey: {self.qkey}")
        return_info.append(f"Attribute: {self.attribute}")
        return_info.append(f"Group: {self.group}")
        return_info.append(f"Question: {self.question}")
        return_info.append(f"Options: {self.options}")
        responses_pretty = [round(r,3) for r in self.responses]
        return_info.append(f"Responses: {responses_pretty}")
        return_info.append(f"Refusal Rate: {self.refusal_rate:.3f}")
        return_info.append(f"Ordinal: {self.ordinal}")
        return_info.append("=" * 57 + "\n")
        return "\n".join(return_info)


class SurveyQuestionCollection:
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = [SurveyQuestion(sample) for sample in samples]
        self.construct_map()

    def construct_map(self) -> None:
        self.subgroup_to_questions: Dict[str, List[str]] = self.construct_stq_map()
        self.question_to_subgroups: Dict[str, List[str]] = self.construct_qts_map()

    def construct_stq_map(self) -> Dict[str, List[str]]:
        stq_map = {}
        for sample in self.samples:
            subgroup = f"{sample.attribute}_{sample.group}"
            if subgroup not in stq_map.keys():
                stq_map[subgroup] = []
            stq_map[subgroup].append(sample.qkey)
        return stq_map
    
    def construct_qts_map(self) -> Dict[str, List[str]]:
        qts_map = {}
        for sample in self.samples:
            if sample.qkey not in qts_map.keys():
                qts_map[sample.qkey] = []
            subgroup = f"{sample.attribute}_{sample.group}"
            qts_map[sample.qkey].append(subgroup)
        return qts_map

    @classmethod
    def from_jsonl(cls,
                   basepath: Union[str, pathlib.Path],
                   filepath: str) -> "SurveyQuestionCollection":
        samples = []
        with open(os.path.join(basepath, filepath), 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format at line {line_number}: {e}")

        return cls(samples)

    def remove_duplicates(self, dims: Union[str,List[str]]) -> None:
        if isinstance(dims, str):
            dims = [dims]
        for dim in dims:
            assert getattr(self.samples[0], dim) is not None, (
                f"--> remove_duplicates(): SurveyQuestion does not have {dim} attr."
            )
        print(f"--> remove_duplicates(): {len(self.samples)} samples before filtering.")
        seen = set()
        unique_samples = []
        for sample in self.samples:
            key = tuple(getattr(sample, dim) for dim in dims)
            if key not in seen:
                seen.add(key)
                unique_samples.append(sample)
        self.samples = unique_samples
        self.construct_map()
        print(f"--> remove_duplicates(): {len(self.samples)} samples left.")

    def filter_by_attribute(self, attribute_meta: Dict[str, List[str]]) -> None:
        print(f"--> filter_by_attribute(): {len(self.samples)} samples before filtering.")
        attr_group_pair = []
        for attr, groups in attribute_meta.items():
            for group in groups:
                attr_group_pair.append((attr, group))
        attr_group_pair = set(attr_group_pair)
        print(f"--> filter_by_attribute(): filtering except for {attr_group_pair}.")
        filtered_samples = []
        for sample in self.samples:
            if (sample.attribute, sample.group) in attr_group_pair:
                filtered_samples.append(sample)
        self.samples = filtered_samples
        self.construct_map()
        print(f"--> filter_by_attribute(): {len(self.samples)} samples left.")

    def save(self, filepath: Union[str, pathlib.Path]) -> None:
        with open(filepath, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(convert_for_json(sample.__dict__)) + "\n")

    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.samples)

    def return_dist_info(self) -> Dict[Tuple, List]:
        dist_info = {}
        for sample in self.samples:
            key = (sample.qkey, (sample.attribute + "_" + sample.group).lower())
            assert key not in dist_info, "Duplicate key found. "
            dist_info[key] = sample.responses
        return dist_info

    def to_list(self) -> List["SurveyQuestion"]:
        return self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self,
                    idx: Union[int, slice]
                    ) -> Union["SurveyQuestion", "SurveyQuestionCollection"]:
        assert isinstance(idx, (int, slice)), (
            f"Indices must be int or slice, not {type(idx)}."
        )
        if isinstance(idx, int):
            return self.samples[idx]
        if isinstance(idx, slice):
            new = self.__class__.__new__(self.__class__)
            new.samples = self.samples[idx]
            new.construct_map()
            return new

    def __iter__(self):
        for sample in self.samples:
            yield sample