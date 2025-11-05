import os
import re
from typing import List, Dict

from sibyl.constants.string_registry_llm import SYSTEM_MESSAGE


def prompt_chat_formatter(prompt: str) -> List[Dict[str, str]]:
    """
    Reformat the plain text formulation of the prompt to chat.
    Chat format is consisted of a system message (SYSTEM_MESSAGE) and
    alternating user-assistant-user-...-user, so total 2n+2 where n is the few-shot examples.
    """
    messages = []
    messages.append({"role": "system", "content": SYSTEM_MESSAGE})
    prompt = prompt.strip()
    parts = re.split(r'(Question:|Answer:)', prompt)
    user_prompts =  [parts[0] + parts[1] + parts[2] + parts[3]]
    assistant_prompts = []
    for _idx in range(4, len(parts)-1, 4):
        assistant_prompts.append(parts[_idx].strip())
        user_prompts.append(parts[_idx+1] + parts[_idx+2] + parts[_idx+3])
    assert len(user_prompts) == len(assistant_prompts) + 1
    for idx in range(len(assistant_prompts)):
        messages.append(
            {"role": "user", "content": user_prompts[idx].strip()}
        )
        messages.append(
            {"role": "assistant", "content": assistant_prompts[idx].strip()}
        )
    messages.append(
        {"role": "user", "content": user_prompts[-1].strip()}
    )
    return messages