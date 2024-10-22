from typing import List, Tuple

import numpy as np
from debate.gen_utils import Conversation, Message, get_len
from models.common import RangeWeight
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

LAST_LINE_INST = ""

START_PREFIX_STD = "These are the selections from other agents: "

START_PREFIX_PROMPT = "These are the selections and confidence values from 1 to 10 (higher means more confident) from other agents: "

END_PREFIX = ""

END_PREFIX_NO_CONF = ""


def construct_message_standard(
    other_agents: List[Conversation],
    conv_idx: int,
) -> Message:
    prefix_string = START_PREFIX_STD

    for agent in other_agents:
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

    prefix_string += END_PREFIX
    return {"role": "user", "content": prefix_string}


def construct_message_prompt(
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
) -> Message:
    prefix_string = START_PREFIX_PROMPT

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution (confidence level is {confidence}): ```{agent_response}```"

    prefix_string += END_PREFIX_NO_CONF
    return {"role": "user", "content": prefix_string}


def construct_message_attention_all(
    this_agent: Conversation,
    this_confidence: float,
    other_agents: List[Conversation],
    other_confidences: List[float],
    conv_idx: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Message:
    if "llama" in tokenizer.name_or_path.lower():
        offset = -1
    elif "mistral" in tokenizer.name_or_path.lower():
        offset = 3
    else:
        raise ValueError("Unknown tokenizer")
    range_weights = []
    this_agent = this_agent.copy()
    len_before_prev = get_len(this_agent[:-1], tokenizer) + 4
    current_len = get_len(this_agent, tokenizer) + offset
    range_weights.append(RangeWeight(len_before_prev, current_len, this_confidence))

    prefix_string = START_PREFIX_STD

    this_agent.append({"role": "user", "content": prefix_string})
    current_len = get_len(this_agent, tokenizer)

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

        this_agent[-1] = {"role": "user", "content": prefix_string}
        new_len = get_len(this_agent, tokenizer)
        range_weights.append(RangeWeight(current_len, new_len, confidence))
        current_len = new_len

    prefix_string += END_PREFIX
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def construct_message_attention_others(
    this_agent: Conversation,
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Message:
    this_agent = this_agent.copy()
    this_agent.append({})

    prefix_string = START_PREFIX_STD

    this_agent[-1] = {"role": "user", "content": prefix_string}
    current_len = get_len(this_agent, tokenizer)

    range_weights = []
    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

        this_agent[-1] = {"role": "user", "content": prefix_string}
        new_len = get_len(this_agent, tokenizer)
        range_weights.append(RangeWeight(current_len, new_len, confidence))
        current_len = new_len

    prefix_string += END_PREFIX
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def gen_question(q_data) -> Tuple[str, str]:
    global LAST_LINE_INST, END_PREFIX, END_PREFIX_NO_CONF
    LAST_LINE_INST = f"The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {''.join([chr(65 + i) for i in range(len(q_data['choices']))])}."
    END_PREFIX = f"""

Can you double check that your response is correct? {LAST_LINE_INST}"""

    END_PREFIX_NO_CONF = f"""

Can you double check that your response is correct? Do not mention your confidence. {LAST_LINE_INST}"""

    question = q_data["question"]
    choices = q_data["choices"]
    answer = q_data["answer"]

    q_str = "Answer the following multiple choice question:\n\n"
    q_str += question + "\n "
    for i, choice in enumerate(choices):
        q_str += f"{chr(65 + i)}. {choice}\n"

    q_str += f"\n\nThink step by step before answering. {LAST_LINE_INST}\n\n"

    return q_str, chr(65 + answer)
