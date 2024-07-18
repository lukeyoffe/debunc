import json
from typing import Dict, List, Tuple, TypedDict

import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import NotRequired


class RangeWeight(TypedDict):
    start: int
    end: int
    weight: float


class Message(TypedDict):
    role: str
    content: str
    uncertainty: NotRequired[float]
    range_weights: NotRequired[List[RangeWeight]]
    confidence: NotRequired[float]


Conversation = List[Message]

Debate = List[Conversation]

Debates = Dict[str, Tuple[Debate, str]]


def get_len(
    context: Conversation, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
):
    return (
        len(
            tokenizer.apply_chat_template(
                context, tokenize=True, add_generation_prompt=True
            )
        )
        - 4
    )


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


START_PREFIX_STD = "These are solutions to the problem from other agents: "
START_PREFIX_PROMPT = "These are solutions and confidence values from 1 to 10 (higher means more confident) to the problem from other agents: "
LAST_LINE_INST = "The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer."


def get_end_prefix_std(question: str) -> str:
    return f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

{LAST_LINE_INST}"""


def get_end_prefix_no_conf(question: str) -> str:
    return f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

Do not mention your confidence. {LAST_LINE_INST}"""


def construct_message_standard(
    question: str,
    other_agents: List[Conversation],
    conv_idx: int,
) -> Message:
    prefix_string = START_PREFIX_STD

    for agent in other_agents:
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

    prefix_string += get_end_prefix_std(question)
    return {"role": "user", "content": prefix_string}


def construct_message_prompt(
    question: str,
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
) -> Message:
    prefix_string = START_PREFIX_PROMPT

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution (confidence level is {confidence}): ```{agent_response}```"

    prefix_string += get_end_prefix_no_conf(question)
    return {"role": "user", "content": prefix_string}


def construct_message_attention_all(
    question: str,
    this_agent: Conversation,
    this_confidence: float,
    other_agents: List[Conversation],
    other_confidences: List[float],
    conv_idx: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Message:
    range_weights = []
    this_agent = this_agent.copy()
    len_before_prev = get_len(this_agent[:-1], tokenizer) + 4
    current_len = get_len(this_agent, tokenizer) + 3
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

    prefix_string += get_end_prefix_std(question)
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def construct_message_attention_others(
    question: str,
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

    prefix_string += get_end_prefix_std(question)
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def format_question(question) -> str:
    formatted_question = f"""
Answer the following math problem. The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer. Think step by step before answering.

{question}
""".strip()

    return formatted_question


class RWJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RangeWeight):
            return obj.__dict__
        return super().default(obj)
