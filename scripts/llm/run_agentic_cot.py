import datetime
import re
import os
import json
import pathlib

from sibyl.utils.logger import start_capture
from sibyl.utils.llm_utils import get_llm_engine, cli_args_parser
from sibyl.constants.string_registry_llm import PROMPT_EXPERT, PROMPT_PREDICT_PREFIX, PROMPT_PREDICT_SUFFIX


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]


def run_baseline_agentic_cot(args) -> None:
    """
    Run inference - Agentic CoT method.
    """
    with open(args.input_path, "r", encoding="utf-8") as f:
        print(f"--> run_baseline_prompting: input path = {args.input_path}")
        lines = [json.loads(line) for line in f if line.strip()]

    sampling_params, llm = get_llm_engine(args, mode="agentic_cot")

    ### Constructing the reflection module inputs
    reflection_module_input = []
    participant_info = []
    question = []
    target_label = []
    for line in lines:
        prompt = line['prompt']
        # context is the individual feature + prior responses.
        context = prompt.rsplit("Question:", 1)[0].strip()
        # indiv_feat (individual feature) is the individual feature part of the context.
        indiv_feat = context.split("Question:")[0].strip().removeprefix("Answer the following question as if your personal information is as follows:").strip()
        # prior_resp is the prior responses part of the context.
        prior_resp = "Question: " + context.split("Question:", 1)[1].strip() if "Question:" in context else "[No prior responses]"
        # q is the target question part.
        q = "Question: " + prompt.rsplit("Question:", 1)[1].strip()
        question.append(q)
        target_label.append(line['label'])
        info = "Participant's information:" + "\n" + indiv_feat + "\n\n" + "Participant's prior responses:" + "\n" + prior_resp
        participant_info.append(info)
        reflection_module_input.append([{"role": "user", "content": info + "\n\n" + PROMPT_EXPERT}])

    output = llm.chat(reflection_module_input, sampling_params)

    ### Constructing the predictor module inputs
    reflection_module_output = []
    for out in output:
        # thinking models generate interal <think> ... </think> thoughts, we extract only the final text.
        text = out.outputs[0].text.split("</think>")[-1].strip()
        reflection_module_output.append(text)

    predictor_module_input = []
    for idx in range(len(lines)):
        prompt = (
            participant_info[idx] + "\n\n"
            + "Expert political scientist's observations/reflections:\n"
            + reflection_module_output[idx] + "\n\n" + "=====" + "\n\n"
            + PROMPT_PREDICT_PREFIX
            + question[idx]
            + PROMPT_PREDICT_SUFFIX
        )
        predictor_module_input.append([{"role": "user", "content": prompt}])

    # adjust max tokens for generation, considering the text generated from reflection module (step 1)
    _input = llm.get_tokenizer().apply_chat_template(predictor_module_input)
    max_len = max([len(tokens) for tokens in _input])
    sampling_params.max_tokens = llm.llm_engine.model_config.max_model_len - max_len
    output = llm.chat(predictor_module_input, sampling_params)

    ### Process the outputs
    correct_cnts = 0
    all_cnts = 0
    agent_outputs = []
    for idx, out in enumerate(output):
        out_str = out.outputs[0].text
        agent_outputs.append(out_str)
        # extract json part
        match = re.search(r"\{.*\}", out_str.lower(), re.DOTALL)
        if match:
            try: # try-except for json decode error due to invalid structure generation
                json_str = match.group(0)
                _data = json.loads(json_str)
                correct = (target_label[idx].lower() == _data.get('response', '').strip()[0])
                correct_cnts += int(correct)
                all_cnts += 1
            except json.JSONDecodeError:
                pass

    print(f"--> agentic CoT framework: accuracy = {correct_cnts}/{len(lines)} = {correct_cnts/len(lines):.4f}") 
    print(f"--> accuracy over valid runs: {correct_cnts}/{all_cnts} = {correct_cnts/all_cnts:.4f}")

    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        args.input_path.split("/")[-1].replace(".jsonl","")
        + f"_llm_agentic_cot_{args.base_model_name_or_path.replace('/','--')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(agent_outputs, f, indent=4)
    print(f"--> run_baseline_agentic_cot: saved to {output_file}")
    

def main():

    args = cli_args_parser()
    if args.use_logger:
        curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _ = start_capture(
            debug=True,
            save_path=os.path.join(ROOT_DIR, "outputs", "logs",
                                   f"llm_inference_{curr_datetime}.log"),
        )

    # check argument consistency
    assert args.input_path is not None and args.output_dir is not None, (
        "input_path and output_dir should be provided."
    )
    assert args.input_path.endswith(".jsonl"), (
        "input_path should be a .jsonl file from the scripts/preprocessing/run_prompt_formulation.py"
    )
    if args.lora_name is None and args.lora_path is not None:
        raise ValueError("LoRA name should be provided when LoRA path provided.")
    
    run_baseline_agentic_cot(args)


if __name__ == "__main__":
    main()
