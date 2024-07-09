import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import json
import random
from typing import List, Tuple

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
from eval_gsm import parse_answer
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
    this_confidence: float,
    other_agents: List[Conversation],
    other_confidences: List[float],
    conv_idx: int,
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


class CustomJSONEncoder(json.JSONEncoder):
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
    questions = questions[:100]

    for eight_shot in [False]:
        filename = f"final/rerun_perfect/gsm_{agents}_{rounds}_{trials}_attention_others_perfect_{ue_method.__class__.__name__}_{eight_shot}shot.json"
        all_trial_data: List[Debates] = []

        current_trial = 0
        if os.path.exists(filename):
            all_trial_data = json.load(open(filename))

        for trial in trange(trials):
            if len(all_trial_data) > trial + 1:
                continue
            if len(all_trial_data) == trial + 1:
                response_dict = all_trial_data[trial]
                if (
                    len(response_dict) > 0
                    and len(list(response_dict.items())[-1][1][0][1]) == 6
                ):
                    current_question = len(response_dict)
                else:
                    current_question = max(0, len(response_dict) - 1)

            if len(all_trial_data) == trial:
                current_question = 0
                response_dict = {}
                all_trial_data.append(response_dict)

            for q_i in trange(
                current_question,
                len(questions),
                initial=current_question,
                total=len(questions),
            ):
                data = questions[q_i]
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
                        confidences = []
                        for agent in agent_contexts:
                            confidences.append(agent[-1]["confidence"])
                    for i, agent_context in enumerate(agent_contexts):
                        if confidences is not None:
                            agent_contexts_other = (
                                agent_contexts[:i] + agent_contexts[i + 1 :]
                            )
                            other_confidences = confidences[:i] + confidences[i + 1 :]
                            message = construct_message(
                                question,
                                this_agent=agent_context,
                                this_confidence=confidences[i],
                                other_agents=agent_contexts_other,
                                other_confidences=other_confidences,
                                conv_idx=2 * round - 1,
                            )
                            agent_context.append(message)

                        completion, _ = generate_answer(agent_context)

                        assistant_message = construct_assistant_message(completion)
                        parsed = parse_answer(completion)
                        gt = float(answer.replace(",", "").split("#### ")[1])

                        assistant_message["confidence"] = 1 if parsed == gt else 1e-5
                        agent_context.append(assistant_message)

                    response_dict[question] = (agent_contexts, answer)
                    all_trial_data[-1] = response_dict
                    json.dump(
                        all_trial_data,
                        open(filename, "w"),
                        cls=CustomJSONEncoder,
                    )
