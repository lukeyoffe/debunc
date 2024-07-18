import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from debate.gen_utils import Conversation, Message, get_len
from models.common import RangeWeight
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

LAST_LINE_INST = "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."

START_PREFIX_STD = "These are solutions to the problem from other agents: "

START_PREFIX_PROMPT = "These are solutions and confidence values from 1 to 10 (higher means more confident) to the problem from other agents: "

END_PREFIX = f"""

Based off the opinion of other agents, can you give an updated response? Think step by step before answering. {LAST_LINE_INST}"""

END_PREFIX_NO_CONF = f"""

Based off the opinion of other agents, can you give an updated response? Do not mention your confidence. Think step by step before answering. {LAST_LINE_INST}"""


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


def format_example(df, idx, include_answer=True):
    choices = ["A", "B", "C", "D"]
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
    if include_answer:
        prompt += f"\nAnswer: {df.iloc[idx, k + 1]}"
    return prompt


def parse_question_answer(df, dev_df, ix, num_shots) -> Tuple[str, str]:
    if num_shots > 0:
        return parse_question_answer_few_shot(df, dev_df, ix, num_shots)
    question = f"""
Answer the following multiple choice question. {LAST_LINE_INST} Think step by step before answering.

{format_example(df, ix, include_answer=False)}
""".strip()

    answer = df.iloc[ix, 5]
    return question, answer


def parse_question_answer_few_shot(df, dev_df, ix, num_shots):
    sep = "\n\n"
    question = f"""
Answer the following multiple choice question.

Examples:

{f"{sep}".join(format_example(dev_df, i) for i in range(num_shots))}

---

YOUR TASK

Answer the following question. Think step by step before answering. {LAST_LINE_INST}

{format_example(df, ix, include_answer=False)}
""".strip()

    answer = df.iloc[ix, 5]
    return question, answer


def get_dfs():
    tasks = json.load(open("george_tasks_glob.json"))
    dfs = [pd.read_csv(task, header=None) for task in tasks]
    dev_dfs = [pd.read_csv(task.replace("test", "dev"), header=None) for task in tasks]
    return tasks, dfs, dev_dfs
