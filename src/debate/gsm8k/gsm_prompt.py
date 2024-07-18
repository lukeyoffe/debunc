import json
import random
from typing import List

import numpy as np
import torch
from debate.gsm8k.common import (
    Conversation,
    Debate,
    Debates,
    Message,
    format_question,
    read_jsonl,
)
from lm_polygraph.estimators import MeanTokenEntropy, TokenSAR
from lm_polygraph.utils.manager import estimate_uncertainty
from models.model import WhiteboxModel
from tqdm import tqdm, trange
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = WhiteboxModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

ue_method = MeanTokenEntropy()


def unc_to_confidence(uncertainties: np.ndarray):
    confidences = 1 / uncertainties
    confidences = confidences * 14 / np.sum(confidences) + 1 / uncertainties.shape[0]
    confidences = np.clip(confidences, 1, 10)
    confidences = np.round(confidences).astype(int)
    return confidences


def construct_message(
    question: str,
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
) -> Message:
    prefix_string = "These are solutions and confidence values from 1 to 10 (higher means more confident) to the problem from other agents: "

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution (confidence level is {confidence}): ```{agent_response}```"

    prefix_string += f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

Do not mention your confidence. The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer."""

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion) -> Message:
    return {"role": "assistant", "content": completion}


def generate_answer(answer_context):
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    out = estimate_uncertainty(model, ue_method, input_text=prompt)
    return out.generation_text, out.uncertainty


if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 5

    random.seed(0)
    questions = read_jsonl("./data/test.jsonl")
    random.shuffle(questions)

    prev_data = [{}]
    all_trial_data: List[Debates] = []
    for trial in trange(trials):
        response_dict: Debates = {}
        all_trial_data.append(response_dict)
        for q_i, data in enumerate(tqdm(questions[:100])):
            question = data["question"]
            answer = data["answer"]
            formatted_question = format_question(question)
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
                    confidences = unc_to_confidence(np.array(uncertainties))
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
                        f"gsm_{agents}_{rounds}_{trials}_prompt_{ue_method.__class__.__name__}.json",
                        "w",
                    ),
                )
