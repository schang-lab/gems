MODEL_NAME_TO_NICKNAME = {
    "meta-llama/Llama-2-7b-hf": "llama2_7b",
    "meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat",
    "meta-llama/Llama-2-13b-hf": "llama2_13b",
    "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat",
    "meta-llama/Llama-2-70b-hf": "llama2_70b",
    "meta-llama/Llama-2-70b-chat-hf": "llama2_70b_chat",
    "meta-llama/Llama-3.1-8B": "llama3p1_8b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3p1_8b_chat",
    "meta-llama/Llama-3.1-70B": "llama3p1_70b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3p1_70b_chat",
    "mistralai/Mistral-7B-v0.1": "mistral_7b",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_chat",
    "mistralai/Mistral-Small-24B-Base-2501": "mistral_24b",
    "mistralai/Mistral-Small-24B-Instruct-2501": "mistral_24b_chat",
    "mistralai/mistralai/Mistral-Small-3.1-24B-Base-2503": "mistral3p1_24b",
    "mistralai/mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral3p1_24b_chat",
    "Qwen/Qwen2.5-32B": "qwen2p5_32b",
    "Qwen/Qwen2.5-32B-Instruct": "qwen2p5_32b_chat",
    "Qwen/Qwen3-32B": "qwen3_32b_chat", # no base model checkpoint
    "Qwen/Qwen3-14B-Base": "qwen3_14b",
    "Qwen/Qwen3-14B": "qwen3_14b_chat",
    "Qwen/Qwen3-8B-Base": "qwen3_8b",
    "Qwen/Qwen3-8B": "qwen3_8b_chat",
    "Qwen/Qwen3-14B-Base": "qwen3_14b",
    "Qwen/Qwen3-14B": "qwen3_14b_chat",
    "Qwen/Qwen3-0.6B-Base": "qwen3_0p6b",
    "Qwen/Qwen3-0.6B": "qwen3_0p6b_chat",
    "openai/gpt-oss-20b": "gpt_oss_20b",
}

MODEL_NICKNAME_TO_NAME = {
    v: k for k, v in MODEL_NAME_TO_NICKNAME.items()
}

FEATURE_PROMPT = """Answer the following question as if your personal information is as follows:
Personal identification number: {id}"""

TRAIT_CODE_TO_TEXT_MAPPING = {
    'age' : 'Age',
    'race' : 'Race or ethnicity',
    'sex' : 'Gender',
    'education' : 'Education level',
    'income' : 'Income level',
    'cregion' : 'Region of residence',
    'relig' : 'Religion',
    'polparty' : 'Political party affiliation',
    'polideology' : 'Political ideology',
    'pre_accuracy' : 'Pre-question expectation of how many of the 20 questions to be answered correctly',
    'pre_percentile' : 'Pre-question expectation of how the participant will perform relative to others, higher the better',
    'pre_average_difficulty' : 'Pre-question expectation of the average difficulty of the questions on a scale of 0 to 10',
    'pre_self_difficulty' : 'Pre-question expectation of the participant’s own difficulty level on a scale of 0 to 10',
    'post_accuracy' : 'Post-question reflection of how many of the 20 questions were answered correctly',
    'post_percentile' : 'Post-question reflection of how the participant performed relative to others, higher the better',
    'post_average_difficulty' : 'Post-question reflection of the average difficulty of the questions on a scale of 0 to 10',
    'post_self_difficulty' : 'Post-question reflection of the participant’s own difficulty level on a scale of 0 to 10',
}

SYSTEM_MESSAGE = (
    "Respond to the following question by choosing one of the available options, "
    + "and strictly answering with the option letter (e.g., 'A', 'B', etc.). "
    + "Do not provide any additional text or explanation."
)

PROMPT_EXPERT = (
    "Imagine you are an expert political scientist (with a PhD) taking notes while observing this content. "
    + "Write observations/reflections about the person's stances about key societal issues. "
    + "(You should make more than 5 observations and fewer than 20. Choose the number that makes sense given the depth of the content above.)"
)

PROMPT_PREDICT_PREFIX = (
    "Task: What you see above is a participant information. "
    + "Based on the information, I want you to predict the participant’s survey responses. "
    + "All questions are multiple choice where you must guess from one of the options presented. "
    + "As you answer, I want you to take the following steps:\n"
    + "Step 1) Describe in a few sentences the kind of person that would choose each of the response options. (”Option Interpretation”)\n"
    + "Step 2) For each response option, reason about why the Participant might answer with the particular option. (”Option Choice”)\n"
    + "Step 3) Write a few sentences reasoning on which of the option best predicts the participant’s response (”Reasoning”)\n"
    + "Step 4) Predict how the participant will actually respond in the survey. Predict based on the information and your thoughts, but ultimately, DON’T overthink it. Use your system 1 (fast, intuitive) thinking. (”Response”)\n\nHere is the question:\n\n"
)

PROMPT_PREDICT_SUFFIX = (
    "\n\n" + "=====" + "\n\n" + "Output format - output your response in json, where you provide the following:\n\n" + """{"Response": "<your predicted response option letter>"}"""
)