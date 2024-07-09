import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import random
from typing import List, Tuple

import numpy as np
import torch
from common import (
    Conversation,
    Debate,
    Debates,
    Message,
    format_question,
    get_len,
    read_jsonl,
)
from lm_polygraph.estimators.token_entropy import MeanTokenEntropy
from lm_polygraph.utils.manager import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.modeling_mistral import RangeWeight
from tqdm import trange
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = WhiteboxModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

ue_method = MeanTokenEntropy()


def construct_message(
    question: str,
    this_agent: Conversation,
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
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

    prefix_string += f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer."""

    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def construct_assistant_message(completion) -> Message:
    return {"role": "assistant", "content": completion}


def generate_answer(answer_context) -> Tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    if "range_weights" in answer_context[-1]:
        model.range_weights = answer_context[-1]["range_weights"]

    out = estimate_uncertainty(model, ue_method, input_text=prompt)
    return out.generation_text, out.uncertainty


class RWJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RangeWeight):
            return obj.__dict__
        return super().default(obj)


if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 5

    random.seed(0)
    questions = read_jsonl("../data/test.jsonl")
    random.shuffle(questions)

    for eight_shot in [False, True]:
        all_trial_data: List[Debates] = []
        for trial in trange(trials):
            response_dict: Debates = {}
            all_trial_data.append(response_dict)
            for q_i, data in enumerate(questions[:100]):
                question = data["question"]
                answer = data["answer"]
                formatted_question = format_question(question, eight_shot)
                agent_contexts: Debate = [
                    [{"role": "user", "content": formatted_question}]
                    for agent in range(agents)
                ]

                for round in range(rounds):
                    torch.cuda.empty_cache()
                    confidences = None
                    if round != 0:
                        uncertainties = []
                        for agent in agent_contexts:
                            agent = agent[-1]
                            uncertainties.append(agent["uncertainty"])
                        confidences = 1 / np.array(uncertainties)
                    for i, agent_context in enumerate(agent_contexts):
                        if confidences is not None:
                            agent_contexts_other = (
                                agent_contexts[:i] + agent_contexts[i + 1 :]
                            )
                            other_confidences = np.concatenate(
                                (confidences[:i], confidences[i + 1 :])
                            )
                            message = construct_message(
                                question,
                                this_agent=agent_context,
                                other_agents=agent_contexts_other,
                                other_confidences=other_confidences,
                                conv_idx=2 * round - 1,
                            )
                            agent_context.append(message)

                        completion, uncertainty = generate_answer(agent_context)

                        assistant_message = construct_assistant_message(completion)
                        assistant_message["uncertainty"] = uncertainty
                        agent_context.append(assistant_message)

                    response_dict[question] = (agent_contexts, answer)
                    all_trial_data[-1] = response_dict
                    json.dump(
                        all_trial_data,
                        open(
                            f"gsm_{agents}_{rounds}_{trials}_attention_{ue_method.__class__.__name__}_{eight_shot}shot_part3.json",
                            "w",
                        ),
                        cls=RWJSONEncoder,
                    )
