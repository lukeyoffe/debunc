from typing import List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from lm_polygraph.utils.modeling_mistral import RangeWeight
from gen_utils import get_len, Message, Conversation


def construct_message_standard(
    other_agents: List[Conversation],
    conv_idx: int,
) -> Message:
    prefix_string = "These are solutions to the problem from other agents: "

    for agent in other_agents:
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

    prefix_string += """

Based off the opinion of other agents, can you provide an updated answer? State the final answer at the end of your response."""
    return {"role": "user", "content": prefix_string}


def construct_message_prompt(
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
) -> Message:
    prefix_string = "These are solutions and confidence values from 1 to 10 (higher means more confident) to the problem from other agents: "

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution (confidence level is {confidence}): ```{agent_response}```"

    prefix_string += """

Based off the opinion of other agents, can you provide an updated answer? State the final answer at the end of your response."""
    return {"role": "user", "content": prefix_string}


def construct_message_prompt_no_conf(
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
) -> Message:
    prefix_string = "These are solutions and confidence values from 1 to 10 (higher means more confident) to the problem from other agents: "

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution (confidence level is {confidence}): ```{agent_response}```"

    prefix_string += """

Based off the opinion of other agents, can you provide an updated answer? Do not mention your confidence. State the final answer at the end of your response."""
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

    prefix_string = "These are solutions to the problem from other agents: "

    this_agent.append({"role": "user", "content": prefix_string})
    current_len = get_len(this_agent, tokenizer)

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

        this_agent[-1] = {"role": "user", "content": prefix_string}
        new_len = get_len(this_agent, tokenizer)
        range_weights.append(RangeWeight(current_len, new_len, confidence))
        current_len = new_len

    prefix_string += """

Based off the opinion of other agents, can you provide an updated answer? State the final answer at the end of your response."""
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

    prefix_string = "These are solutions to the problem from other agents: "

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

    prefix_string += """

Based off the opinion of other agents, can you provide an updated answer? State the final answer at the end of your response."""
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def gen_question() -> Tuple[str, int]:
    a, b, c, d = np.random.randint(0, 30, size=4)
    question = f"What is the result of {a}+{b}*{c}+{d}? State the final answer at the end of your response."
    answer = int(a + b * c + d)
    return question, answer
