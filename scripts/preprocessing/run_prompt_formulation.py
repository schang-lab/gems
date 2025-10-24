from __future__ import annotations
import argparse
import os
import json
import pathlib
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from sibyl.graph.node import NodeCollection
from sibyl.graph.entity import Individual, Identity
from sibyl.utils.surveydata_utils import RawDataManager
from sibyl.config.dataset_map import DATASET_MAP
from sibyl.constants.string_registry_llm import TRAIT_CODE_TO_TEXT_MAPPING, FEATURE_PROMPT

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
SAVING_FORMAT = "{base_filename}_topk_{topk}_{split}.jsonl"

class EmbeddingManager:

    def __init__(self, embedding_path: str):
        """
        Load the precomputed text embeddings in dictionary {question key: embedding (Tensor)}.
        """
        self.embedding_path = embedding_path
        self.embeddings = torch.load(embedding_path, map_location='cpu')

    def return_rankings(self,
                        qkeys_query: List[str],
                        qkeys_example: List[str],
                        top_k: int) -> List[List[str]]:
        """
        Semantics:
            Given a list of query question keys and example question keys,
            return the top_k most similar example question keys for each query (except itself)
            based on cosine similarity of their embeddings
            in the increasing order of similarity (most similar last).
        """
        assert len(qkeys_query) > 0 and len(qkeys_example) > 0 and top_k > 0, (
            "qkeys_query and qkeys_example must be non-empty. top_k must be positive.\n"
            f"len(qkeys_query)={len(qkeys_query)}, len(qkeys_example)={len(qkeys_example)}, top_k={top_k}."
        )
        target_is_example = (set(qkeys_query) == set(qkeys_example))
        assert top_k <= len(qkeys_example) - int(target_is_example), (
            "top_k is larger than the number of available example questions."
        )
        sim_matrix = cosine_similarity(
            [self.embeddings[qkey] for qkey in qkeys_query],
            [self.embeddings[qkey] for qkey in qkeys_example],
        )
        rankings_dict = {}
        for i, qkey in enumerate(qkeys_query):
            sim_scores = sim_matrix[i]
            if target_is_example:
                top_indices = sim_scores.argsort()[-top_k-1:-1]
            else:
                top_indices = sim_scores.argsort()[-top_k:]
            rankings_dict[qkey] = [qkeys_example[j] for j in top_indices]
            assert qkey not in rankings_dict[qkey], "Self found in top-k, should not happen"
        return rankings_dict


def _format_q(q_str: str, options: Dict[str, str]) -> str:
    """
    Format the question string and option texts into an input prompt.
    """
    repr_str = f"Question: {q_str}\n"
    for idx, opt_str in enumerate(options.values()):
        repr_str += chr(ord('A') + idx) + ". " + opt_str + "\n"
    return repr_str + "\nAnswer:"


def _textify_response(response, survey_manager) -> Tuple[str, str]:
    """
    Take the response {qkey: choice_value}, convert to question and choice strings.
    """
    qkey, choice = response
    q_str : str= survey_manager.qstrings[qkey]
    options : Dict[float, str] = survey_manager.fetch_options(qkey)
    repr_str : str = _format_q(q_str, options)
    choice_str : str = chr(ord('A') + list(options.keys()).index(choice))
    choice_str += ". " + options[choice]
    return repr_str, choice_str


def _textify_indiv_feature(id: Identity, key: float) -> str:
    """
    Take the individual feature information and convert it to a text string.
    """
    repr_str = FEATURE_PROMPT.format(id=str(key))
    for trait, attr in id.attributes.items():
        attr_ucase = attr[0].upper() + attr[1:]
        repr_str += f"\n{TRAIT_CODE_TO_TEXT_MAPPING[trait]}: {attr_ucase}"
    return repr_str


def _textify_individual(individual: Individual,
                        survey_manager: RawDataManager,
                        embedding_manager: EmbeddingManager,
                        top_k: int,
                        context_info: Dict[str, float],
                        target_info: Dict[str, float]) -> List[str]:
    """
    Per-individual textification.
    """
    text_feat = _textify_indiv_feature(id=individual.identity, key=individual.id)
    context_qkeys = list(context_info.keys())
    text_context = [_textify_response(resp, survey_manager) for resp in context_info.items()]
    target_qkeys = list(target_info.keys())
    text_target = [_textify_response(resp, survey_manager) for resp in target_info.items()]

    if top_k == 0: # no few-shot examples
        return [(text_feat + "\n\n" + text_t[0], text_t[1][0])
                for text_t in text_target]
    
    qkey_rankings: Dict[str, List[str]] = embedding_manager.return_rankings(
        qkeys_query=target_qkeys,
        qkeys_example=context_qkeys,
        top_k=top_k,
    )
    fewshot_examples = {
        qkey: text_context[_idx][0] + " " + text_context[_idx][1]
        for _idx, qkey in enumerate(context_qkeys)
    }
    return_list = []
    for _idx, qkey in enumerate(target_qkeys):
        text_t = text_target[_idx]
        sim_qkeys = qkey_rankings[qkey]
        fewshot_str = "\n\n".join(fewshot_examples[sqk] for sqk in sim_qkeys)
        return_list.append((text_feat + "\n\n" + fewshot_str + "\n\n" + text_t[0], text_t[1][0]))
    return return_list


def _textify_all(individual_dict: Dict[float, Individual],
                 survey_manager: RawDataManager,
                 embedding_manager: EmbeddingManager,
                 top_k: int,
                 example_info: Dict[str, Dict[str, float]],
                 target_info: Dict[str, Dict[str, float]]) -> List[str]:
    texts = []
    for id_str, target_qdict in tqdm(target_info.items(), desc="Textifying individuals"):
        id = float(id_str)
        assert id in individual_dict, f"Individual {id} not found in the dataset."
        indiv_data = _textify_individual(individual_dict[id],
                                         survey_manager,
                                         embedding_manager,
                                         top_k,
                                         example_info.get(id_str, {}),
                                         target_qdict)
        texts.extend(indiv_data)         
    return texts


def main():

    parser = argparse.ArgumentParser(description="Get text embeddings.")
    parser.add_argument("--dataset", type=str,
                        choices=DATASET_MAP.keys(), required=True,
                        help="Input dataset.")
    parser.add_argument("--top_k", type=int,
                        default=0,
                        help="Number of few-shot examples to include in the input prompt. (0 for zero-shot)")
    parser.add_argument("--text_embedding_path", type=str,
                        default="outputs/text_embeddings/opinionqa_text_embeddings_gemini-embedding-001.pth",
                        help="Path to the text embedding file.")
    parser.add_argument("--split_path", type=str,
                        default="outputs/dataset_splits/opinionqa_individual_val0p05_test0p60_evalpartial_0p40_seed42.jsonl",
                        help="Path to the dataset split file.")
    args = parser.parse_args()
    dataset_info = DATASET_MAP[args.dataset]

    nodecollection = NodeCollection(
        source=dataset_info['source'],
        basepath=DATA_DIR / args.dataset,
        qkeypath=dataset_info['qkeypath'],
        qstringpath=dataset_info['qstringpath'],
        valid_traits=dataset_info['valid_traits'],
        use_selective_subgroups=dataset_info['use_selective_subgroups'],
        min_questions_per_indiv=dataset_info['min_questions_per_indiv'],
        dataset_name=args.dataset,
    )
    individual_dict, survey_manager = nodecollection.individuals, nodecollection.manager
    embedding_manager = EmbeddingManager(os.path.join(ROOT_DIR, args.text_embedding_path))
    top_k = args.top_k
    assert top_k >= 0, "top_k must be non-negative."
    input_path = args.split_path
    input_data = json.load(open(os.path.join(ROOT_DIR, input_path), 'r'))

    samples = {}
    for split_name in ["train", "val", "test"]:
        samples[split_name] = _textify_all(individual_dict,
                                           survey_manager,
                                           embedding_manager,
                                           top_k,
                                           input_data['train'], input_data[split_name])
    
    for split_name in ["train", "val", "test"]:
        save_path = os.path.join(
            ROOT_DIR, "outputs", "llm_prompts",
            SAVING_FORMAT.format(
                base_filename=args.split_path.split("/")[-1].replace(".jsonl", ""),
                topk=top_k,
                split=split_name
            )
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            for prompt, label in samples[split_name]:
                f.write(json.dumps({"prompt": prompt, "label": label}) + "\n")


if __name__ == "__main__":
    main()